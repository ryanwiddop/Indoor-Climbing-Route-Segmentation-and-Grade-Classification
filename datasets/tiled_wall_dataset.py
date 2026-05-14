import logging
import os
import random

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw, ImageOps
from pycocotools.coco import COCO
from torch.utils.data import Dataset

LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), "logs/tiled_hold_dataset.log")
if not os.path.exists(os.path.dirname(LOG_FILE_PATH)):
    os.makedirs(os.path.dirname(LOG_FILE_PATH))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(LOG_FILE_PATH)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False


class TiledWallDataset(Dataset):
    def __init__(self, img_dir, coco_json, augment=False, transform=None):
        self.img_dir = img_dir
        self.augment = augment
        self.transform = transform
        self.coco = COCO(coco_json)

        self.image_ids = sorted(self.coco.imgs.keys())

        cats = sorted(self.coco.loadCats(self.coco.getCatIds()), key=lambda c: c["id"])
        self.hold_type_to_idx = {c["name"]: i + 1 for i, c in enumerate(cats)}
        self.idx_to_hold_type = {v: k for k, v in self.hold_type_to_idx.items()}
        self.cat_id_to_label = {c["id"]: i + 1 for i, c in enumerate(cats)}

        n_pos = sum(1 for i in self.image_ids if self.coco.getAnnIds(imgIds=i))
        logger.info(
            "TiledWallDataset ready: %d tiles (%d positive, %d negative) | "
            "%d hold classes: %s",
            len(self.image_ids), n_pos, len(self.image_ids) - n_pos,
            len(self.hold_type_to_idx), list(self.hold_type_to_idx.keys()),
        )

    def __len__(self):
        return len(self.image_ids)

    @staticmethod
    def _rasterize(xs, ys, W, H):
        mask = Image.new("L", (W, H), 0)
        ImageDraw.Draw(mask).polygon(list(zip(xs, ys)), fill=1)
        return np.array(mask, dtype=np.uint8)

    def _empty_target(self, index, H, W):
        return {
            "boxes": torch.empty((0, 4), dtype=torch.float32),
            "labels": torch.empty((0,), dtype=torch.int64),
            "masks": torch.empty((0, H, W), dtype=torch.uint8),
            "area": torch.empty((0,), dtype=torch.float32),
            "image_id": torch.tensor([index]),
        }

    def __getitem__(self, index):
        img_id = self.image_ids[index]
        info = self.coco.loadImgs(img_id)[0]
        path = os.path.join(self.img_dir, info["file_name"])
        image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        W, H = image.size
        image_tensor = self.transform(image) if self.transform else F.to_tensor(image)

        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))

        boxes, labels, masks, areas = [], [], [], []
        for ann in anns:
            seg = ann.get("segmentation")
            if not seg or not isinstance(seg, list) or not seg[0]:
                continue
            poly = seg[0]
            if len(poly) < 6:
                continue
            xs = list(poly[::2])
            ys = list(poly[1::2])

            x_min = max(0, min(xs))
            y_min = max(0, min(ys))
            x_max = min(W, max(xs))
            y_max = min(H, max(ys))
            if x_max <= x_min or y_max <= y_min:
                logger.warning(
                    "%s:%s - degenerate bbox after clipping, skipping.",
                    info["file_name"], ann["id"],
                )
                continue

            label = self.cat_id_to_label.get(ann["category_id"])
            if label is None:
                logger.warning(
                    "%s:%s - category_id %s not in mapping; skipping.",
                    info["file_name"], ann["id"], ann["category_id"],
                )
                continue

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(label)
            masks.append(self._rasterize(xs, ys, W, H))
            areas.append((x_max - x_min) * (y_max - y_min))

        if not boxes:
            return image_tensor, self._empty_target(index, H, W)

        boxes_t = torch.tensor(boxes, dtype=torch.float32)
        masks_t = torch.from_numpy(np.stack(masks, axis=0))

        if self.augment:
            if random.random() < 0.5:
                image_tensor = F.hflip(image_tensor)
                masks_t = torch.flip(masks_t, dims=[2])
                boxes_t[:, [0, 2]] = W - boxes_t[:, [2, 0]]
            if random.random() < 0.5:
                image_tensor = F.vflip(image_tensor)
                masks_t = torch.flip(masks_t, dims=[1])
                boxes_t[:, [1, 3]] = H - boxes_t[:, [3, 1]]

        target = {
            "boxes": boxes_t,
            "labels": torch.tensor(labels, dtype=torch.int64),
            "masks": masks_t,
            "area": torch.tensor(areas, dtype=torch.float32),
            "image_id": torch.tensor([index]),
        }
        return image_tensor, target


def collate_fn(batch):
    batch = [x for x in batch if x is not None]
    return tuple(zip(*batch)) if batch else ([], [])
