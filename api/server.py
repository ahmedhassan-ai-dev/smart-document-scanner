from fastapi import FastAPI, UploadFile, File
from typing import List
import numpy as np
import cv2
import os

from c



































ore.detector import detect_document_from_image
from engine.enhancement import enhance_all
from services.image_service import save_image
from services.pdf_service import images_to_pdf

app = FastAPI()

# تأكد إن فولدر outputs موجود
OUTPUT_DIR = "outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)



#  1. Scan Single Image
@app.post("/scan")
async def scan(file: UploadFile = File(...)):

    try:
        contents = await file.read()

        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return {"error": "Invalid image file"}

        # detect document
        doc = detect_document_from_image(image)

        if doc is None:
            return {"error": "No document detected"}

        # apply filters
        results = enhance_all(doc)

        paths = {}

        for name, img in results.items():
            path = save_image(img, prefix=name)
            paths[name] = path

        return {
            "status": "success",
            "results": paths
        }

    except Exception as e:
        return {"error": str(e)}


#  2. Export PDF from selected images
@app.post("/export/pdf")
def export_pdf(images: List[str]):

    try:
        if not images or len(images) == 0:
            return {"error": "No images provided"}

        pdf_path = images_to_pdf(images)

        return {
            "status": "success",
            "pdf": pdf_path
        }

    except Exception as e:
        return {"error": str(e)}



#  3. Multi-Page Scan → Direct PDF
@app.post("/scan/multi")
async def scan_multi(files: List[UploadFile]):

    try:
        if not files or len(files) == 0:
            return {"error": "No files uploaded"}

        paths = []

        for file in files:
            contents = await file.read()

            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                continue

            doc = detect_document_from_image(image)

            if doc is not None:
                path = save_image(doc, prefix="page")
                paths.append(path)

        if len(paths) == 0:
            return {"error": "No valid documents detected"}

        pdf_path = images_to_pdf(paths)

        return {
            "status": "success",
            "pdf": pdf_path
        }

    except Exception as e:
        return {"error": str(e)}



#  Health Check
@app.get("/")
def home():
    return {"message": "Document Scanner API is running "}