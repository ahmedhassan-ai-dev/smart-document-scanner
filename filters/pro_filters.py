import cv2
import numpy as np


def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ===============================
# model1: Boosted / Enhanced
# ===============================
def model1_enhanced(img):
    return cv2.convertScaleAbs(img, alpha=1.3, beta=20)


# ===============================
# model2: Strong Scan
# ===============================
def model2_strong(img):
    g = gray(img)
    blur = cv2.GaussianBlur(g, (5,5), 0)
    return cv2.adaptiveThreshold(
        blur,255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,11,2)


# ===============================
# model3: B&W Scan
# ===============================
def model3_bw(img):
    g = gray(img)
    _,bw = cv2.threshold(g,0,255,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw


# ===============================
# model4: OCR Mode
# ===============================
def model4_ocr(img):
    g = gray(img)
    return cv2.adaptiveThreshold(
        g,255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,15,5)


# ===============================
# model5: Premium Scanner
# ===============================
def model5_premium(img):
    g = gray(img)
    clahe = cv2.createCLAHE(3,(8,8))
    x = clahe.apply(g)
    return cv2.adaptiveThreshold(
        x,255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,15,8)


# ===============================
# model6: Doc White
# ===============================
def model6_docwhite(img):
    g = gray(img)
    blur = cv2.GaussianBlur(g,(35,35),0)
    norm = cv2.divide(g,blur,scale=255)
    return cv2.adaptiveThreshold(
        norm,255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,21,10)


# ===============================
# model7: Ultra Scan
# ===============================
def model7_ultra(img):
    g = gray(img)
    blur = cv2.GaussianBlur(g,(25,25),0)
    norm = cv2.divide(g,blur,scale=255)
    return cv2.adaptiveThreshold(
        norm,255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,21,10)


# ===============================
# model8: Sharp Text / Paper Enhance
# ===============================
def model8_sharp(img):
    g = gray(img)
    kernel = np.array([[0,-1,0],
                       [-1,5,-1],
                       [0,-1,0]])
    return cv2.filter2D(g,-1,kernel)


# ===============================
# model9: HD Scan
# ===============================
def model9_hd(img):
    return cv2.detailEnhance(img, sigma_s=20, sigma_r=0.2)