import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk

from core.detector import detect_document_from_image
from engine.enhancement import enhance_all


class ScannerApp:

    def __init__(self, root):
        self.root = root
        self.root.title("📄 Smart Scanner")
        self.root.geometry("1100x750")

        self.results = {}
        self.selected_image = None
        self.pages = []   # 📚 الصفحات

        # =========================
        # 🔘 Buttons
        # =========================
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="📂 Upload Image", command=self.load_image).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="🔙 Back", command=self.go_back).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="➕ Add Page", command=self.add_page).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="📄 Export PDF", command=self.export_pdf).pack(side=tk.LEFT, padx=5)

        # عدد الصفحات
        self.pages_label = tk.Label(root, text="Pages: 0")
        self.pages_label.pack()

        # =========================
        # 🖼 Preview Image
        # =========================
        self.canvas = tk.Label(root)
        self.canvas.pack()

        # =========================
        # 📸 Filters Grid
        # =========================
        self.grid_frame = tk.Frame(root)
        self.grid_frame.pack(pady=10)

    # =========================
    # 📂 Load Image
    # =========================
    def load_image(self):
        path = filedialog.askopenfilename()

        if not path:
            return

        image = cv2.imread(path)

        if image is None:
            messagebox.showerror("Error", "Image not found")
            return

        doc = detect_document_from_image(image)

        if doc is None:
            messagebox.showerror("Error", "No document detected")
            return

        self.results = enhance_all(doc)
        self.show_grid()

    # =========================
    # 🖼 عرض الفلاتر كصور
    # =========================
    def show_grid(self):

        for widget in self.grid_frame.winfo_children():
            widget.destroy()

        row, col = 0, 0

        for name, img in self.results.items():

            frame = tk.Frame(self.grid_frame, bd=2, relief="ridge")
            frame.grid(row=row, column=col, padx=5, pady=5)

            display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            display = cv2.resize(display, (150, 200))

            display = Image.fromarray(display)
            display = ImageTk.PhotoImage(display)

            label_img = tk.Label(frame, image=display)
            label_img.image = display
            label_img.pack()

            tk.Label(frame, text=name).pack()

            label_img.bind("<Button-1>", lambda e, im=img: self.select_image(im))

            col += 1
            if col == 5:
                col = 0
                row += 1

    # =========================
    # 👆 اختيار صورة
    # =========================
    def select_image(self, img):
        self.selected_image = img
        self.show_image(img)

    # =========================
    # 🖼 عرض صورة كبيرة
    # =========================
    def show_image(self, img):
        display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        display = cv2.resize(display, (500, 700))

        display = Image.fromarray(display)
        display = ImageTk.PhotoImage(display)

        self.canvas.configure(image=display)
        self.canvas.image = display

    # =========================
    # ➕ Add Page
    # =========================
    def add_page(self):
        if self.selected_image is None:
            messagebox.showwarning("Warning", "No image selected")
            return

        self.pages.append(self.selected_image.copy())
        self.pages_label.config(text=f"Pages: {len(self.pages)}")

        messagebox.showinfo("Added", f"Page added! Total: {len(self.pages)}")

    # =========================
    # 📄 Export PDF
    # =========================
    def export_pdf(self):

        if not self.pages:
            messagebox.showwarning("Warning", "No pages added")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf")]
        )

        if not path:
            return

        from PIL import Image

        pil_images = []

        for img in self.pages:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_images.append(Image.fromarray(rgb))

        try:
            pil_images[0].save(
                path,
                save_all=True,
                append_images=pil_images[1:]
            )
            messagebox.showinfo("Success", f"PDF saved at:\n{path}")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    # =========================
    # 🔙 Back (Step by Step)
    # =========================
    def go_back(self):

        # لو فاتح صورة → ارجع للفلاتر
        if self.selected_image is not None:
            self.selected_image = None
            self.canvas.configure(image="")
            self.canvas.image = None
            return

        # لو في فلاتر → امسحهم
        if self.results:
            for widget in self.grid_frame.winfo_children():
                widget.destroy()

            self.results = {}
            return


# =========================
# ▶ Run App
# =========================
if __name__ == "__main__":
    root = tk.Tk()
    app = ScannerApp(root)
    root.mainloop()
