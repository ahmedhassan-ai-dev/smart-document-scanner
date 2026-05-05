from filters.pro_filters import *

def enhance_all(img):

    return {
        "model1_enhanced": model1_enhanced(img),
        "model2_strong": model2_strong(img),
        "model3_bw": model3_bw(img),
        "model4_ocr": model4_ocr(img),
        "model5_premium": model5_premium(img),
        "model6_docwhite": model6_docwhite(img),
        "model7_ultra": model7_ultra(img),
        "model8_sharp": model8_sharp(img),
        "model9_hd": model9_hd(img)
    }