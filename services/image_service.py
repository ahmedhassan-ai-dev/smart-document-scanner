import cv2
import os
import uuid

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_image(img, ext=".jpg"):
    """
    Save image safely (supports JPG / PNG)
    """

    # ✅ توحيد الامتدادات
    if ext.lower() in [".jpeg", "jpeg"]:
        ext = ".jpg"

    if not ext.startswith("."):
        ext = "." + ext

    # ✅ تحويل grayscale إلى BGR (علشان JPG)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    filename = f"{uuid.uuid4()}{ext}"
    path = os.path.join(OUTPUT_DIR, filename)

    # ✅ حفظ الصورة
    success = cv2.imwrite(path, img)

    if not success:
        raise Exception("❌ Failed to save image")

    return path


def save_multiple_images(images, ext=".jpg"):
    """
    Save list of images
    """
    paths = []

    for img in images:
        path = save_image(img, ext)
        paths.append(path)

    return paths
