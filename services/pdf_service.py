from PIL import Image
import os

def images_to_pdf(image_paths, output_path="output.pdf"):

    if not image_paths:
        print("❌ No images provided")
        return None

















































































    images = []

    for path in image_paths:
        if not os.path.exists(path):
            print(f"❌ Image not found: {path}")
            continue

        img = Image.open(path)

        # مهم جدًا
        if img.mode != "RGB":
            img = img.convert("RGB")

        images.append(img)

    if len(images) == 0:
        print("❌ No valid images")
        return None

    try:
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:]
        )

        print(f"✅ PDF saved at: {output_path}")
        return output_path

    except Exception as e:
        print("❌ Error while saving PDF:", e)
        return None