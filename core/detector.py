# core/detector.py

import cv2
import numpy as np
from rembg import remove

from core.transform import perspective_transform
from config import RESIZE_HEIGHT, KERNEL_SIZE, DEBUG


def detect_document_from_image(image):

    # ==========================================
    # Resize
    # ==========================================
    ratio = image.shape[0] / RESIZE_HEIGHT
    orig = image.copy()

    image = cv2.resize(image, (
        int(image.shape[1] / ratio),
        RESIZE_HEIGHT
    ))

    # ==========================================
    # AI Mask
    # ==========================================
    try:
        mask = remove(image, only_mask=True)
    except:
        return None

    # ==========================================
    # Threshold + Morphology
    # ==========================================
    _, th = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    kernel = np.ones(KERNEL_SIZE, np.uint8)

    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)

    # ==========================================
    # Contours
    # ==========================================
    contours, _ = cv2.findContours(
        th,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    c = max(contours, key=cv2.contourArea)

    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)

    screenCnt = None

    # Dynamic approximation
    for eps in np.linspace(0.01, 0.05, 10):
        approx = cv2.approxPolyDP(hull, eps * peri, True)

        if len(approx) == 4:
            screenCnt = approx
            break

    # fallback
    if screenCnt is None:
        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect)
        screenCnt = np.array(box, dtype="int32").reshape(4, 1, 2)

    # ==========================================
    # Perspective Transform
    # ==========================================
    warped = perspective_transform(
        orig,
        screenCnt.reshape(4, 2) * ratio
    )

    return warped