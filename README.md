# 📄 Smart Document Scanner

## 🚀 Project Overview

Smart Document Scanner using AI + Computer Vision

* Detect document automatically
* Crop & perspective transform
* Apply multiple enhancement filters
* Export as Image or PDF
* Multi-page PDF support

---

## 🧠 Tech Stack

### Backend

* Python
* OpenCV
* rembg (AI segmentation)
* FastAPI

### Frontend

* Desktop (Tkinter)
* Streamlit App 

---

## ✨ Features

* 📷 Auto Document Detection
* 🧾 Scan Enhancement (9 Models)
* 📑 Export as PDF
* 🗂 Multi-image PDF
* 🔄 Image Sorting
* 💾 Save as Image

---

## ⚙️ Run Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn server:app --reload
```

---

## 💻 Run Desktop App

```bash
python app.py
```

---

## 🖥️ Streamlit Web App

This project includes a full interactive web interface built with **Streamlit**.

### ✨ What you can do:
- Upload single or multiple images
- Preview original & processed images
- Apply multiple filters
- Compare outputs
- Export results as PDF

### ▶️ Run the Streamlit App

```bash
pip install -r requirements.txt
streamlit run app.py
---

