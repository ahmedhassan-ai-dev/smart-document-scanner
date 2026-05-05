# main.py

import cv2
import os
from core.detector import detect_document_from_image
from engine.enhancement import enhance_all
from services.image_service import save_image
from services.pdf_service import images_to_pdf

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def process_single_image(path):
    image = cv2.imread(path)

    if image is None:
        print("❌ Image not found")
        return None

    doc = detect_document_from_image(image)

    if doc is None:
        print("❌ No document detected")
        return None

    results = enhance_all(doc)

    return results


def show_results(results):
    for name, img in results.items():
        print("📌 Showing:", name)
        cv2.imshow(name, img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def choose_and_save(results):
    print("\nAvailable Filters:")
    keys = list(results.keys())

    for i, key in enumerate(keys):
        print(f"{i} - {key}")

    choice = int(input("Choose filter number: "))
    selected = keys[choice]

    img = results[selected]

    path = save_image(img)

    print(f"✅ Saved: {path}")
    return path


def multi_to_pdf():
    print("\n📚 Multi Image to PDF Mode")

    paths = []

    while True:
        img_path = input("Enter image path (or 'done'): ")

        if img_path.lower() == "done":
            break

        results = process_single_image(img_path)

        if results:
            saved = choose_and_save(results)
            paths.append(saved)

    if len(paths) == 0:
        print("❌ No images to convert")
        return

    pdf_path = images_to_pdf(paths)

    print(f"📄 PDF Saved: {pdf_path}")


def main_menu():
    while True:
        print("\n====== 📄 SMART SCANNER ======")
        print("1 - Scan Image")
        print("2 - Multi Scan → PDF")
        print("3 - Exit")

        choice = input("Choose option: ")

        if choice == "1":
            path = input("Enter image path: ")

            results = process_single_image(path)

            if results:
                show_results(results)

                save = input("Save result? (y/n): ")

                if save.lower() == "y":
                    choose_and_save(results)

        elif choice == "2":
            multi_to_pdf()

        elif choice == "3":
            print("👋 Exit")
            break

        else:
            print("❌ Invalid choice")


if __name__ == "__main__":
    main_menu()