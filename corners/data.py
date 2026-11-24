"""Runway-crop dataset: one sample per runway label, cropped around a (detector or jittered ground-truth) box."""
import os, sys
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lard_geom import corners_of, DATA  # noqa: E402

CROP = 256          # network input size
STRIDE = 4          # heatmap = 64 x 64
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
FLIP = [1, 0, 3, 2]  # TL,TR,BR,BL -> after horizontal flip: TR,TL,BL,BR

# Runway keypoints as (u, v) on the unit square: u across the runway (left -> right), v along it (far end -> threshold).
# First 4 = corners TL, TR, BR, BL; then 2 points on each short edge and 4 on each long edge (16 total).
CORNER_UV = [(0, 0), (1, 0), (1, 1), (0, 1)]
EDGE_UV = ([(1 / 3, 0), (2 / 3, 0)] + [(1, v) for v in (0.2, 0.4, 0.6, 0.8)] +
           [(2 / 3, 1), (1 / 3, 1)] + [(0, v) for v in (0.8, 0.6, 0.4, 0.2)])
KP_UV = {4: CORNER_UV, 16: CORNER_UV + EDGE_UV}


def flip_perm(uv):
    """Index permutation for a horizontal flip (u -> 1 - u)."""
    return [min(range(len(uv)), key=lambda j: abs(uv[j][0] - (1 - u)) + abs(uv[j][1] - v)) for u, v in uv]


def keypoints_from_corners(corners, n_kp=4):
    """Image positions of the n_kp runway keypoints from the 4 labelled corners (exact: plane-to-image homography)."""
    if n_kp == 4:
        return corners.astype(np.float32)
    H = cv2.getPerspectiveTransform(np.float32(CORNER_UV), corners.astype(np.float32))
    return cv2.perspectiveTransform(np.float32(KP_UV[n_kp])[None], H)[0]


def crop_box(box, margin=1.6, min_side=40.0):
    """Square crop (cx, cy, side) around a box (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = box
    side = max(margin * max(x1 - x0, y1 - y0) + 12, min_side)
    return (x0 + x1) / 2, (y0 + y1) / 2, side


def affine(cx, cy, side):
    """2x3 matrix mapping full-image px -> crop px."""
    s = CROP / side
    return np.array([[s, 0, CROP / 2 - s * cx], [0, s, CROP / 2 - s * cy]], np.float32)


class RunwayCrops(Dataset):
    """rows: list of label dicts (must contain image + corners). boxes: optional dict row_index -> detector box.
    train=True: jitter the ground-truth box, photometric jitter, random horizontal flip."""

    def __init__(self, rows, boxes=None, train=False, seed=0, n_kp=4):
        self.rows, self.boxes, self.train, self.n_kp = rows, boxes or {}, train, n_kp
        self.rng = np.random.default_rng(seed)
        self.flip = FLIP if n_kp == 4 else flip_perm(KP_UV[n_kp])

    def __len__(self):
        return len(self.rows)

    def _box(self, i, gt):
        if i in self.boxes:
            return self.boxes[i]
        x0, y0 = gt.min(0); x1, y1 = gt.max(0)
        w, h = x1 - x0 + 2, y1 - y0 + 2
        r = self.rng if self.train else np.random.default_rng(i)          # fixed jitter for val
        dx, dy = r.normal(0, 0.12, 2) * [w, h]; sc = np.exp(r.normal(0, 0.12, 2))
        cx, cy = (x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy
        return cx - w * sc[0] / 2, cy - h * sc[1] / 2, cx + w * sc[0] / 2, cy + h * sc[1] / 2

    def __getitem__(self, i):
        row = self.rows[i]
        img = cv2.cvtColor(cv2.imread(os.path.join(DATA, row['image'])), cv2.COLOR_BGR2RGB)
        gt = corners_of(row)
        cx, cy, side = crop_box(self._box(i, gt), margin=self.rng.uniform(1.4, 2.0) if self.train else 1.6)
        A = affine(cx, cy, side)
        crop = cv2.warpAffine(img, A, (CROP, CROP), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        kp = keypoints_from_corners(gt, self.n_kp) @ A[:, :2].T + A[:, 2]
        if self.train:
            if self.rng.random() < 0.5:
                crop = crop[:, ::-1].copy(); kp = kp[self.flip]; kp[:, 0] = CROP - 1 - kp[:, 0]
            crop = np.clip(crop.astype(np.float32) * self.rng.uniform(0.75, 1.25) + self.rng.uniform(-20, 20), 0, 255)
        x = (crop.astype(np.float32) / 255 - MEAN) / STD
        return (torch.from_numpy(x.transpose(2, 0, 1).copy()),
                torch.from_numpy(kp.astype(np.float32)),
                torch.from_numpy(A), i)
