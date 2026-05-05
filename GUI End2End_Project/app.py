import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

def order_points(pts):
    rect = np.zeros((4, 2), dtype='float32')
    s = pts.sum(axis=1); rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1); rect[1] = pts[np.argmin(diff)]; rect[3] = pts[np.argmax(diff)]
    return rect

def perspective_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    maxW = int(max(np.linalg.norm(br-bl), np.linalg.norm(tr-tl)))
    maxH = int(max(np.linalg.norm(tr-br), np.linalg.norm(tl-bl)))
    if maxW < 10 or maxH < 10: return image
    dst = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], dtype='float32')
    return cv2.warpPerspective(image, cv2.getPerspectiveTransform(rect, dst), (maxW, maxH))

def detect_document(image):
    from rembg import remove
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    small = cv2.resize(image, (int(image.shape[1]/ratio), 500))
    mask = remove(small, only_mask=True)
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((9,9), np.uint8)
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel)
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel)
    cnts, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < small.shape[0]*small.shape[1]*0.05: return None
    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)
    screenCnt = None
    for eps in np.linspace(0.01, 0.1, 30):
        approx = cv2.approxPolyDP(hull, eps*peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break
    if screenCnt is None:
        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect)
        screenCnt = np.array(box, dtype="int32").reshape(4,1,2)
    return perspective_transform(orig, screenCnt.reshape(4,2)*ratio)


def smart_crop(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape)==3 else image.copy()
    h_img, w_img = gray.shape[:2]
    _, white_mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    k = np.ones((15,15), np.uint8)
    white_mask = cv2.morphologyEx(cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)
    cnts, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        largest = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(largest) > h_img*w_img*0.15:
            x,y,w,h = cv2.boundingRect(largest)
            pad = 5
            y1,y2 = max(0,y-pad), min(h_img,y+h+pad)
            x1,x2 = max(0,x-pad), min(w_img,x+w+pad)
            cropped = image[y1:y2, x1:x2]
            if cropped.shape[0]*cropped.shape[1] >= h_img*w_img*0.2:
                return cropped
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
    coords = cv2.findNonZero(binary)
    if coords is None: return image
    x,y,w,h = cv2.boundingRect(coords); pad=10
    cropped = image[max(0,y-pad):min(h_img,y+h+pad), max(0,x-pad):min(w_img,x+w+pad)]
    return cropped if cropped.shape[0]*cropped.shape[1] >= h_img*w_img*0.2 else image

def to_gray(img): return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
def pure_white(img): _,bw=cv2.threshold(to_gray(img),180,255,cv2.THRESH_BINARY); return bw
def dark_text(img): return cv2.convertScaleAbs(to_gray(img),alpha=1.5)
def smooth_scan(img): return cv2.GaussianBlur(to_gray(img),(5,5),0)
def ocr_mode(img): return cv2.adaptiveThreshold(to_gray(img),255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
def notebook_mode(img): return cv2.createCLAHE(4,(8,8)).apply(to_gray(img))
def sharp_text(img): return cv2.filter2D(to_gray(img),-1,np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]))
def warm_paper(img): x=img.copy(); x[:,:,2]=cv2.add(x[:,:,2],20); return x
def cool_paper(img): x=img.copy(); x[:,:,0]=cv2.add(x[:,:,0],20); return x
def pencil_scan(img): g=to_gray(img); return cv2.divide(g,255-cv2.GaussianBlur(255-g,(21,21),0),scale=256)
def hd_scan(img): return cv2.detailEnhance(img,sigma_s=20,sigma_r=0.2)
def text_bold(img): _,bw=cv2.threshold(to_gray(img),160,255,cv2.THRESH_BINARY); return cv2.erode(bw,np.ones((2,2),np.uint8),iterations=1)
def high_contrast(img): return cv2.convertScaleAbs(to_gray(img),alpha=1.8)
def light_paper(img): return cv2.convertScaleAbs(to_gray(img),alpha=1,beta=35)
def anti_shadow(img): g=to_gray(img); return cv2.divide(g,cv2.GaussianBlur(g,(51,51),0),scale=255)

def premium_mode(img): x=cv2.createCLAHE(3,(8,8)).apply(to_gray(img)); return cv2.adaptiveThreshold(x,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,8)
def doc_white(img): g=to_gray(img); n=cv2.divide(g,cv2.GaussianBlur(g,(35,35),0),scale=255); return cv2.adaptiveThreshold(n,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,10)
def ultra_clean(img): _,bw=cv2.threshold(to_gray(img),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU); return bw
def ink_boost(img): g=to_gray(img); return cv2.convertScaleAbs(g-0.8*cv2.Laplacian(g,cv2.CV_64F))
def baseline_enhance(img): return cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(to_gray(img))
def calc_sharpness(i):
    if len(i.shape)==3: i=cv2.cvtColor(i,cv2.COLOR_BGR2GRAY)
    return round(cv2.Laplacian(i,cv2.CV_64F).var(),2)
def calc_contrast(i):
    if len(i.shape)==3: i=cv2.cvtColor(i,cv2.COLOR_BGR2GRAY)
    return round(i.std(),2)
def calc_brightness(i):
    if len(i.shape)==3: i=cv2.cvtColor(i,cv2.COLOR_BGR2GRAY)
    return round(i.mean(),2)
def to_rgb(i):
    if len(i.shape)==2: return cv2.cvtColor(i,cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(i,cv2.COLOR_BGR2RGB)
def to_pil(i):
    if len(i.shape)==2: return Image.fromarray(i).convert("RGB")
    return Image.fromarray(cv2.cvtColor(i,cv2.COLOR_BGR2RGB))
ALL_FILTERS={"Sharp Text":sharp_text,"High Contrast":high_contrast,"Anti Shadow":anti_shadow,"Premium Scanner":premium_mode,"Doc White":doc_white,"Pure White":pure_white,"Dark Text":dark_text,"OCR Mode":ocr_mode,"Notebook Mode":notebook_mode,"Ink Boost":ink_boost,"Smooth Scan":smooth_scan,"Warm Paper":warm_paper,"Cool Paper":cool_paper,"Pencil Scan":pencil_scan,"HD Scan":hd_scan,"Text Bold":text_bold,"Light Paper":light_paper,"Ultra Clean":ultra_clean}

st.set_page_config(page_title="Document Scanner Pro",page_icon="\U0001f4c4",layout="wide")
if "processed" not in st.session_state: st.session_state["processed"]=False
with st.sidebar:
    st.title("\U0001f4c4 Control Panel"); st.markdown("---")
    selected_filters=st.multiselect("\U0001f3a8 Select Filters:",list(ALL_FILTERS.keys()),default=["Sharp Text","High Contrast","Anti Shadow","Premium Scanner","Doc White"])
    st.markdown("---")
    dark_mode=st.toggle("\U0001f319 Dark Mode",value=False)
    skip_detection=st.toggle("Skip Detection",value=False)
    do_smart_crop=st.toggle("\u2702\ufe0f Smart Crop",value=True,help="Crop to paper only")
    st.markdown("---")
    for i,s in enumerate(["Detection","Smart Crop","Enhancement","PDF"],1): st.markdown(f"**{i}.** {s}")
if dark_mode:
    st.markdown('<style>.main-header{background:linear-gradient(135deg,#2d3748,#1a202c);padding:2rem;border-radius:15px;margin-bottom:2rem;text-align:center;border:1px solid #4a5568}.main-header h1{color:#e2e8f0;font-size:2.5rem}.main-header p{color:#a0aec0}.step-badge{display:inline-block;background:#4a5568;color:#63b3ed;padding:.3rem .8rem;border-radius:20px;font-size:.85rem;font-weight:600;margin-bottom:.5rem}.metric-card{background:#2d3748;padding:1.5rem;border-radius:12px;text-align:center;border:1px solid #4a5568}.metric-card h3{color:#e2e8f0}.metric-card p{color:#a0aec0}.metric-card .value{font-size:2rem;font-weight:bold;color:#63b3ed}.metric-card .delta-positive{color:#68d391}.metric-card .delta-negative{color:#fc8181}div[data-testid="stSidebar"]{background:linear-gradient(180deg,#1a202c,#2d3748)}.stApp{background:#1a202c}</style>',unsafe_allow_html=True)
else:
    st.markdown('<style>.main-header{background:linear-gradient(135deg,#a8edea,#fed6e3);padding:2rem;border-radius:15px;margin-bottom:2rem;text-align:center}.main-header h1{color:#2d3748;font-size:2.5rem}.main-header p{color:#4a5568}.step-badge{display:inline-block;background:linear-gradient(135deg,#89f7fe,#66a6ff);color:#1a365d;padding:.3rem .8rem;border-radius:20px;font-size:.85rem;font-weight:600;margin-bottom:.5rem}.metric-card{background:linear-gradient(135deg,#fdfcfb,#e2d1c3);padding:1.5rem;border-radius:12px;text-align:center;border:1px solid #f0e6dc}.metric-card h3{color:#4a5568}.metric-card .value{font-size:2rem;font-weight:bold;color:#38b2ac}.metric-card .delta-positive{color:#48bb78}.metric-card .delta-negative{color:#f56565}div[data-testid="stSidebar"]{background:linear-gradient(180deg,#fdfcfb,#e8f4f8)}.stApp{background:linear-gradient(180deg,#fff,#f7fafc)}</style>',unsafe_allow_html=True)
st.markdown('<div class="main-header"><h1>Document Scanner Pro</h1><p>Upload - Detect - Crop - Enhance - PDF</p></div>',unsafe_allow_html=True)

tab_upload,tab_results,tab_compare,tab_pdf=st.tabs(["Upload","Results","Metrics","PDF"])
with tab_upload:
    c1,c2=st.columns([1,1])
    with c1:
        st.subheader("Upload Documents")
        uploaded_files=st.file_uploader("Drag and drop",type=["jpg","jpeg","png","bmp","webp"],accept_multiple_files=True)
        if uploaded_files: st.success(f"{len(uploaded_files)} file(s)"); process_btn=st.button("Process All",type="primary",use_container_width=True)
    with c2:
        if uploaded_files:
            pcols=st.columns(min(len(uploaded_files),4))
            for i,f in enumerate(uploaded_files):
                with pcols[i%4]: st.image(Image.open(f),caption=f.name,use_container_width=True); f.seek(0)
    if uploaded_files and process_btn:
        if not selected_filters: st.error("Pick filters.")
        else:
            all_results=[]; progress=st.progress(0)
            for idx,uf in enumerate(uploaded_files):
                progress.progress(int((idx/len(uploaded_files))*100),text=f"{idx+1}/{len(uploaded_files)}")
                original=cv2.imdecode(np.asarray(bytearray(uf.read()),dtype=np.uint8),cv2.IMREAD_COLOR)
                if original is None: continue
                warped=detect_document(original) if not skip_detection else None
                if warped is None: warped=original.copy()
                if do_smart_crop: warped=smart_crop(warped)
                bl=baseline_enhance(warped); fres={fn:ALL_FILTERS[fn](warped) for fn in selected_filters}
                m={"sharpness_baseline":calc_sharpness(bl),"contrast_baseline":calc_contrast(bl),"brightness_baseline":calc_brightness(bl)}
                for fn,fi in fres.items(): m[f"sharpness_{fn}"]=calc_sharpness(fi); m[f"contrast_{fn}"]=calc_contrast(fi); m[f"brightness_{fn}"]=calc_brightness(fi)
                all_results.append({"name":uf.name,"original":original,"warped":warped,"baseline":bl,"filters":fres,"metrics":m})
            progress.progress(100,text="Done!"); st.session_state.update({"all_results":all_results,"selected_filters":selected_filters,"processed":True})
            st.balloons(); st.success(f"{len(all_results)} done!")

with tab_results:
    if st.session_state.get("processed") and st.session_state.get("all_results"):
        ar=st.session_state["all_results"]; names=[r["name"] for r in ar]
        ci=st.selectbox("Image:",range(len(names)),format_func=lambda i:f"{i+1}. {names[i]}") if len(names)>1 else 0
        r=ar[ci]; c1,c2=st.columns(2)
        with c1: st.markdown('<div class="step-badge">CROPPED</div>',unsafe_allow_html=True); st.image(to_rgb(r["warped"]),use_container_width=True)
        with c2: st.markdown('<div class="step-badge">BASELINE</div>',unsafe_allow_html=True); st.image(to_rgb(r["baseline"]),use_container_width=True)
        st.markdown("---")
        fn_list=list(r["filters"].keys())
        for rs in range(0,len(fn_list),3):
            cols=st.columns(3)
            for j,fn in enumerate(fn_list[rs:rs+3]):
                with cols[j]: st.markdown(f'<div class="step-badge">{fn}</div>',unsafe_allow_html=True); st.image(to_rgb(r["filters"][fn]),use_container_width=True)
    else: st.info("Upload and process images first.")
with tab_compare:
    if st.session_state.get("processed") and st.session_state.get("all_results"):
        ar=st.session_state["all_results"]; names=[r["name"] for r in ar]
        ci=st.selectbox("Image:",range(len(names)),format_func=lambda i:f"{i+1}. {names[i]}",key="cmp") if len(names)>1 else 0
        r=ar[ci]; m=r["metrics"]
        for fn in r["filters"]:
            st.markdown(f"#### {fn}"); c1,c2,c3=st.columns(3)
            for col,lbl,k in [(c1,"Sharpness","sharpness"),(c2,"Contrast","contrast"),(c3,"Brightness","brightness")]:
                b=m[f"{k}_baseline"]; o=m.get(f"{k}_{fn}",0); d=round(o-b,2); dc="delta-positive" if d>0 else "delta-negative"; s="+" if d>0 else ""
                with col: st.markdown(f'<div class="metric-card"><h3>{lbl}</h3><p>Baseline: <strong>{b}</strong></p><p class="value">{o}</p><p class="{dc}">{s}{d}</p></div>',unsafe_allow_html=True)
            st.markdown("---")
    else: st.info("Upload and process images first.")
with tab_pdf:
    if st.session_state.get("processed") and st.session_state.get("all_results"):
        ar=st.session_state["all_results"]; sf=st.session_state.get("selected_filters",[])
        pf=st.selectbox("Filter:",["Baseline"]+sf,key="pdf_f"); pages=[]
        for r in ar:
            fi=r["filters"].get(pf) if pf!="Baseline" else None
            pages.append({"image":to_pil(fi if fi is not None else r["baseline"]),"name":r["name"]})
        for rs in range(0,len(pages),4):
            cols=st.columns(4)
            for j,p in enumerate(pages[rs:rs+4]):
                with cols[j]: st.image(p["image"],caption=p["name"],use_container_width=True)
        if st.button("Export PDF",type="primary",use_container_width=True):
            imgs=[p["image"].convert("RGB") for p in pages]; buf=io.BytesIO()
            if len(imgs)==1: imgs[0].save(buf,format="PDF")
            else: imgs[0].save(buf,format="PDF",save_all=True,append_images=imgs[1:])
            buf.seek(0); st.download_button("Download PDF",data=buf.getvalue(),file_name=f"scan_{pf.replace(chr(32),chr(95)).lower()}.pdf",mime="application/pdf",use_container_width=True)
    else: st.info("Upload and process images first.")
