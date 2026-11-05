"""Read image and semantic-mask pairs directly from the official archives."""

from io import BytesIO
from pathlib import PurePosixPath
import runpy

import numpy as np
from PIL import Image, ImageOps
import torch


class SegmentationArchive:
    def __init__(self, images, masks, split, resize, mapping, num_classes):
        self.images = images
        self.masks = masks
        self.resize = resize
        self.mapping = mapping
        self.num_classes = num_classes
        self.image_paths = self._paths(images, split, ".jpg")
        self.mask_paths = self._paths(masks, split, ".png")
        self.ids = sorted(self.image_paths.keys() & self.mask_paths.keys())
        if not self.ids:
            raise ValueError(f"No paired JPEGs and PNGs in archive split {split!r}.")

    @staticmethod
    def _paths(archive, split, suffix):
        paths = (PurePosixPath(name) for name in archive.namelist())
        return {p.stem: str(p) for p in paths if p.parent.name == split and p.suffix == suffix}

    def target(self, image_id):
        with Image.open(BytesIO(self.masks.read(self.mask_paths[image_id]))) as source:
            mask = np.asarray(source)
        if mask.ndim != 2 or mask.dtype != np.uint8:
            raise ValueError(f"Expected an indexed PNG mask for {image_id}.")
        return self.mapping[mask]

    def load(self, image_id):
        target = self.target(image_id)
        with Image.open(BytesIO(self.images.read(self.image_paths[image_id]))) as source:
            image = ImageOps.exif_transpose(source)
            # Some JPEG EXIF rotations disagree with the orientation of the paired mask.
            if image.size[::-1] != target.shape and source.size[::-1] == target.shape:
                image = source
            image = np.asarray(image.convert("RGB"))
        if image.shape[:2] != target.shape:
            raise ValueError(f"The image and mask dimensions differ for {image_id}.")
        height, width = target.shape
        image = self.resize.get_transform(image).apply_image(image)
        item = {"image": torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))),
                "height": height, "width": width}
        return item, target


def ade150(archive, resize):
    mapping = np.full(256, 255, dtype=np.uint8)
    mapping[1:151] = np.arange(150, dtype=np.uint8)
    result = SegmentationArchive(archive, archive, "validation", resize, mapping, 150)
    if len(result.ids) != 2000:
        raise ValueError("Expected all 2,000 labeled ADE20K A-150 validation images.")
    return result


def coco_stuff(images, masks, sed_root, resize):
    # Use the released SED label IDs, including COCO's gaps between categories.
    categories = runpy.run_path(str(sed_root / "datasets" / "prepare_coco_stuff.py"))["COCO_CATEGORIES"]
    mapping = np.full(256, 255, dtype=np.uint8)
    for category in categories:
        mapping[category["id"]] = category["trainId"]
    return SegmentationArchive(images, masks, "val2017", resize, mapping, len(categories))


def sample_labeled(dataset, size, rng):
    selected = []
    for image_id in rng.sample(dataset.ids, len(dataset.ids)):
        if np.any(dataset.target(image_id) != 255):
            selected.append(image_id)
            if len(selected) == size:
                return selected
    raise ValueError(f"Requested {size} calibration images and found {len(selected)} with valid labels.")
