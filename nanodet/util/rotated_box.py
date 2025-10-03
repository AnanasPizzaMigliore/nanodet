"""Utilities for manipulating rotated bounding boxes."""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np


def rbox2poly(rboxes: np.ndarray) -> np.ndarray:
    """Convert rotated boxes to polygons.

    Args:
        rboxes (np.ndarray): Rotated boxes in ``(cx, cy, w, h, angle_deg)`` format.

    Returns:
        np.ndarray: Polygons with shape ``(N, 4, 2)``.
    """

    if rboxes.size == 0:
        return rboxes.reshape(0, 4, 2)

    cx = rboxes[:, 0]
    cy = rboxes[:, 1]
    w = rboxes[:, 2]
    h = rboxes[:, 3]
    angle = np.deg2rad(rboxes[:, 4])

    cos = np.cos(angle)
    sin = np.sin(angle)

    w_half = w / 2.0
    h_half = h / 2.0

    # Corner points before rotation relative to center.
    corners = np.stack(
        [
            np.stack([-w_half, -h_half], axis=1),
            np.stack([w_half, -h_half], axis=1),
            np.stack([w_half, h_half], axis=1),
            np.stack([-w_half, h_half], axis=1),
        ],
        axis=1,
    )

    rot = np.stack(
        [
            np.stack([cos, -sin], axis=1),
            np.stack([sin, cos], axis=1),
        ],
        axis=1,
    )
    # Apply rotation then translation.
    polys = corners @ rot
    polys[..., 0] += cx[:, None]
    polys[..., 1] += cy[:, None]
    return polys


def poly2rbox(polygons: np.ndarray) -> np.ndarray:
    """Convert 4-point polygons to rotated boxes.

    Args:
        polygons (np.ndarray): Polygons with shape ``(N, 4, 2)``.

    Returns:
        np.ndarray: Rotated boxes in ``(cx, cy, w, h, angle_deg)`` format.
    """

    if polygons.size == 0:
        return polygons.reshape(0, 5)

    polygons = polygons.astype(np.float32)
    rboxes = []
    for poly in polygons:
        rect = cv2.minAreaRect(poly)
        (cx, cy), (w, h), angle = rect
        # cv2.minAreaRect returns angle in [-90, 0) with w >= h by default.
        if w < h:
            w, h = h, w
            angle += 90.0
        angle = ((angle + 180.0) % 180.0) - 90.0
        rboxes.append([cx, cy, w, h, angle])
    return np.asarray(rboxes, dtype=np.float32)


def rbox2bbox(rboxes: np.ndarray) -> np.ndarray:
    """Convert rotated boxes to axis-aligned bounding boxes."""

    if rboxes.size == 0:
        return rboxes.reshape(0, 4)

    cx = rboxes[:, 0]
    cy = rboxes[:, 1]
    w = rboxes[:, 2]
    h = rboxes[:, 3]
    angle = np.deg2rad(rboxes[:, 4])

    cos = np.abs(np.cos(angle))
    sin = np.abs(np.sin(angle))

    bbox_w = cos * w + sin * h
    bbox_h = sin * w + cos * h

    x1 = cx - bbox_w / 2.0
    y1 = cy - bbox_h / 2.0
    x2 = cx + bbox_w / 2.0
    y2 = cy + bbox_h / 2.0
    return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)


def warp_rboxes(
    rboxes: np.ndarray,
    matrix: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Apply a perspective transform to rotated boxes."""

    if rboxes.size == 0:
        return rboxes

    polys = rbox2poly(rboxes)
    ones = np.ones((polys.shape[0] * 4, 3), dtype=np.float32)
    ones[:, :2] = polys.reshape(-1, 2)
    warped = ones @ matrix.T
    warped = warped[:, :2] / warped[:, 2:3]
    warped = warped.reshape(-1, 4, 2)

    warped[..., 0] = np.clip(warped[..., 0], 0, width)
    warped[..., 1] = np.clip(warped[..., 1], 0, height)

    return poly2rbox(warped)


def ensure_rboxes(data: Iterable[Iterable[float]]) -> np.ndarray:
    """Convert a Python iterable to a rotated box ndarray."""

    return np.asarray(list(data), dtype=np.float32).reshape(-1, 5)
