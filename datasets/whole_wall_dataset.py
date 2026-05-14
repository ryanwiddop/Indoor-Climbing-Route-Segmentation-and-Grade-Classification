import logging
import os

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw, ImageOps
from pycocotools.coco import COCO
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class WholeWallDataset(Dataset):
    def __init__(self, img_dir, coco_json, hold_type_to_idx=None, return_masks=False, transform=None):
        self.img_dir = img_dir
        self.return_masks = return_masks
        self.transform = transform
        self.coco = COCO(coco_json)

        cats = sorted(self.coco.loadCats(self.coco.getCatIds()), key=lambda c: c["id"])
        self.cat_id_to_name = {c["id"]: c["name"] for c in cats}

        if hold_type_to_idx is None:
            self.hold_type_to_idx = {c["name"]: i + 1 for i, c in enumerate(cats)}
        else:
            self.hold_type_to_idx = dict(hold_type_to_idx)
        self.idx_to_hold_type = {v: k for k, v in self.hold_type_to_idx.items()}

        self.image_ids = []
        for img_id in sorted(self.coco.imgs.keys()):
            info = self.coco.imgs[img_id]
            if os.path.isfile(os.path.join(img_dir, info["file_name"])):
                self.image_ids.append(img_id)
            else:
                logger.warning(
                    "Image %s in COCO but not in %s - skipping.",
                    info["file_name"], img_dir,
                )

    def __len__(self):
        return len(self.image_ids)

    @staticmethod
    def _rasterize(xs, ys, W, H):
        mask = Image.new("L", (W, H), 0)
        ImageDraw.Draw(mask).polygon(list(zip(xs, ys)), fill=1)
        return np.array(mask, dtype=np.uint8)

    def __getitem__(self, index):
        img_id = self.image_ids[index]
        info = self.coco.loadImgs(img_id)[0]
        fname = info["file_name"]
        path = os.path.join(self.img_dir, fname)
        image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        W, H = image.size
        image_tensor = self.transform(image) if self.transform else F.to_tensor(image)

        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))

        boxes, labels, polygons = [], [], []
        hold_types, route_ids, route_grades, is_volumes = [], [], [], []
        masks_np = []

        for ann in anns:
            seg = ann.get("segmentation")
            if not seg or not isinstance(seg, list) or not seg[0]:
                continue
            poly = seg[0]
            if len(poly) < 6:
                continue
            xs = list(poly[::2])
            ys = list(poly[1::2])

            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            if x_max <= x_min or y_max <= y_min:
                logger.warning("%s:%s - degenerate bbox, skipping.", fname, ann["id"])
                continue

            cat_name = self.cat_id_to_name.get(ann["category_id"])
            label = self.hold_type_to_idx.get(cat_name)
            if label is None:
                logger.warning(
                    "%s:%s - category %r not in mapping %s; skipping.",
                    fname, ann["id"], cat_name, sorted(self.hold_type_to_idx.keys()),
                )
                continue

            attr = ann.get("attributes", {}) or {}
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(label)
            polygons.append((xs, ys))
            hold_types.append(cat_name)
            route_ids.append(attr.get("route_id", ""))
            route_grades.append(attr.get("route_grade", ""))
            is_volumes.append(cat_name == "volume")

            if self.return_masks:
                masks_np.append(self._rasterize(xs, ys, W, H))

        if boxes:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
            area_t = (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            area_t = torch.zeros((0,), dtype=torch.float32)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "area": area_t,
            "image_id": torch.tensor([index]),
            "polygons": polygons,
            "filename": fname,
            "image_size": (W, H),
            "hold_types": hold_types,
            "route_ids": route_ids,
            "route_grades": route_grades,
            "is_volumes": is_volumes,
        }

        if self.return_masks:
            if masks_np:
                target["masks"] = torch.from_numpy(np.stack(masks_np, axis=0))
            else:
                target["masks"] = torch.zeros((0, H, W), dtype=torch.uint8)

        return image_tensor, target


def collate_fn(batch):
    batch = [x for x in batch if x is not None]
    return tuple(zip(*batch)) if batch else ([], [])
