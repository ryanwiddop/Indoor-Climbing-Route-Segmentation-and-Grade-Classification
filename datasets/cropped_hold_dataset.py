import os
import logging
import random

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from pycocotools.coco import COCO

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torch.utils.data import Dataset

LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), "logs/cropped_hold_dataset.log")
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

MIN_LAB_PIXELS = 5
ERODE_FRAC = 0.1


class CroppedHoldDataset(Dataset):
    def __init__(self, img_dir, coco_json, size=224, augment=False, vflip=False,
                 transform=None):
        self.img_dir = img_dir
        self.size = size
        self.augment = augment
        self.vflip = vflip
        self.transform = transform
        self.coco = COCO(coco_json)

        self.image_ids = sorted(self.coco.imgs.keys())

        cats = sorted(self.coco.loadCats(self.coco.getCatIds()), key=lambda c: c["id"])
        self.hold_type_to_idx = {c["name"]: i + 1 for i, c in enumerate(cats)}
        self.idx_to_hold_type = {v: k for k, v in self.hold_type_to_idx.items()}
        self.cat_id_to_label = {c["id"]: i + 1 for i, c in enumerate(cats)}

        logger.info(
            "CroppedHoldDataset ready: %d holds | %dx%d | %d hold classes: %s",
            len(self.image_ids), size, size,
            len(self.hold_type_to_idx), list(self.hold_type_to_idx.keys()),
        )

    def __len__(self):
        return len(self.image_ids)

    @staticmethod
    def _rasterize(xs, ys, W, H):
        mask = Image.new("L", (W, H), 0)
        ImageDraw.Draw(mask).polygon(list(zip(xs, ys)), fill=1)
        return np.array(mask, dtype=np.uint8)

    @staticmethod
    def _erode(mask):
        h, w = mask.shape
        k = int(round(ERODE_FRAC * min(h, w)))
        if k < 1:
            return mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        eroded = cv2.erode(mask, kernel)
        return eroded if eroded.sum() >= MIN_LAB_PIXELS else mask

    @staticmethod
    def _lab_stats(rgb, mask):
        sel = mask > 0
        if sel.sum() < MIN_LAB_PIXELS:
            return (float("nan"),) * 3
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        L = lab[:, :, 0][sel].astype(np.float32)
        a = lab[:, :, 1][sel].astype(np.float32)
        b = lab[:, :, 2][sel].astype(np.float32)

        chroma = np.sqrt((a - 128.0) ** 2 + (b - 128.0) ** 2)
        chromatic = chroma >= np.percentile(chroma, 75)
        if chromatic.sum() < MIN_LAB_PIXELS:
            chromatic = np.ones_like(chromatic)

        return float(np.median(L)), float(np.median(a[chromatic])), float(np.median(b[chromatic]))

    def _letterbox(self, image, mask):
        w, h = image.size
        scale = self.size / max(w, h)
        new_w = max(1, min(self.size, int(round(w * scale))))
        new_h = max(1, min(self.size, int(round(h * scale))))

        image = image.resize((new_w, new_h), Image.BILINEAR)
        mask = Image.fromarray(mask).resize((new_w, new_h), Image.NEAREST)

        canvas = Image.new("RGB", (self.size, self.size), (0, 0, 0))
        mask_canvas = Image.new("L", (self.size, self.size), 0)
        left = (self.size - new_w) // 2
        top = (self.size - new_h) // 2
        canvas.paste(image, (left, top))
        mask_canvas.paste(mask, (left, top))
        return canvas, np.array(mask_canvas, dtype=np.uint8)

    def __getitem__(self, index):
        img_id = self.image_ids[index]
        info = self.coco.loadImgs(img_id)[0]
        path = os.path.join(self.img_dir, info["file_name"])
        image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        W, H = image.size

        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        if not anns:
            logger.warning("%s - no annotation on crop, skipping.", info["file_name"])
            return None
        ann = anns[0]
        if len(anns) > 1:
            logger.warning(
                "%s - %d annotations on crop, using id %s.",
                info["file_name"], len(anns), ann["id"],
            )

        label = self.cat_id_to_label.get(ann["category_id"])
        if label is None:
            logger.warning(
                "%s - category_id %s not in mapping; skipping.",
                info["file_name"], ann["category_id"],
            )
            return None

        seg = ann.get("segmentation")
        poly = seg[0] if seg and isinstance(seg, list) and seg[0] else None
        if poly is None or len(poly) < 6:
            logger.warning(
                "%s - unusable polygon, falling back to full crop as mask.",
                info["file_name"],
            )
            mask = np.ones((H, W), dtype=np.uint8)
        else:
            mask = self._rasterize(list(poly[::2]), list(poly[1::2]), W, H)
            if mask.sum() == 0:
                logger.warning(
                    "%s - polygon rasterized empty, falling back to full crop as mask.",
                    info["file_name"],
                )
                mask = np.ones((H, W), dtype=np.uint8)

        aspect = info.get("source_aspect", H / W)

        if self.augment:
            if random.random() < 0.5:
                image = F.hflip(image)
                mask = mask[:, ::-1].copy()
            if self.vflip and random.random() < 0.5:
                image = F.vflip(image)
                mask = mask[::-1].copy()
            if random.random() < 0.5:
                image = T.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                )(image)

        med_L, med_a, med_b = self._lab_stats(np.array(image), self._erode(mask))

        image, mask = self._letterbox(image, mask)
        image_tensor = self.transform(image) if self.transform else F.to_tensor(image)

        if self.augment and random.random() < 0.5:
            image_tensor = T.GaussianBlur(kernel_size=random.choice([3, 5, 7]))(image_tensor)

        target = {
            "label": torch.tensor(label, dtype=torch.int64),
            "mask": torch.from_numpy(mask).unsqueeze(0),
            "lab": torch.tensor([med_L, med_a, med_b], dtype=torch.float32),
            "aspect": torch.tensor(aspect, dtype=torch.float32),
            "image_id": torch.tensor([index]),
        }
        return image_tensor, target


def collate_fn(batch):
    batch = [x for x in batch if x is not None]
    if not batch:
        return [], {}
    images, targets = zip(*batch)
    return torch.stack(images, 0), {
        k: torch.stack([t[k] for t in targets], 0) for k in targets[0]
    }
