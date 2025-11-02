import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
import time
from tqdm import tqdm

from .models import SVGPyTorchConverter
from .utils import ImageUtils, QualityValidator, SVGGenerator


class SVGPyTorchGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PyTorch SVG Converter - Professional")
        self.root.geometry("1200x800")

        # Initialize converter
        self.converter = SVGPyTorchConverter()

        # Variables
        self.image_path = None
        self.original_image = None
        self.processed_image = None
        self.output_size = tk.IntVar(value=512)
        self.conversion_method = tk.StringVar(value="canny")
        self.quality_setting = tk.StringVar(value="high")

        self.setup_ui()

    def setup_ui(self):
        # Configure main grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Create main notebook (tabbed interface)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Main conversion tab
        self.conversion_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.conversion_frame, text="SVG Conversion")

        # Settings tab
        self.settings_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.settings_frame, text="Settings & Benchmark")

        self.setup_conversion_tab()
        self.setup_settings_tab()

    def setup_conversion_tab(self):
        # File selection
        file_frame = ttk.LabelFrame(self.conversion_frame, text="Image Selection", padding="10")
        file_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=5)
        file_frame.columnconfigure(0, weight=1)

        self.file_entry = ttk.Entry(file_frame)
        self.file_entry.grid(row=0, column=0, sticky="ew", padx=(0, 10))

        ttk.Button(file_frame, text="Browse",
                   command=self.browse_image).grid(row=0, column=1)

        # Conversion settings
        settings_frame = ttk.LabelFrame(self.conversion_frame, text="Conversion Settings", padding="10")
        settings_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=5)

        # Size control
        ttk.Label(settings_frame, text="Output Size:").grid(row=0, column=0, sticky="w")
        ttk.Scale(settings_frame, from_=64, to=2048, variable=self.output_size,
                  orient=tk.HORIZONTAL, command=self.on_size_change).grid(row=0, column=1, sticky="ew", padx=10)
        self.size_label = ttk.Label(settings_frame, text="512 px")
        self.size_label.grid(row=0, column=2)

        # Method selection
        ttk.Label(settings_frame, text="Method:").grid(row=1, column=0, sticky="w", pady=5)
        methods = [("Canny Edge", "canny"), ("Color Segmentation", "segmentation"),
                   ("Adaptive Threshold", "adaptive"), ("Vector Quantization", "vector_quant")]

        for i, (text, value) in enumerate(methods):
            ttk.Radiobutton(settings_frame, text=text, variable=self.conversion_method,
                            value=value, command=self.on_method_change).grid(row=1, column=i + 1, padx=5)

        # Preview frames
        preview_frame = ttk.Frame(self.conversion_frame)
        preview_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=10)
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.columnconfigure(1, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        self.conversion_frame.rowconfigure(2, weight=1)

        # Original preview
        orig_frame = ttk.LabelFrame(preview_frame, text="Original Image")
        orig_frame.grid(row=0, column=0, sticky="nsew", padx=5)
        orig_frame.columnconfigure(0, weight=1)
        orig_frame.rowconfigure(0, weight=1)

        self.orig_canvas = tk.Canvas(orig_frame, bg='white', relief=tk.SUNKEN)
        self.orig_canvas.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Processed preview
        proc_frame = ttk.LabelFrame(preview_frame, text="Processed Image")
        proc_frame.grid(row=0, column=1, sticky="nsew", padx=5)
        proc_frame.columnconfigure(0, weight=1)
        proc_frame.rowconfigure(0, weight=1)

        self.proc_canvas = tk.Canvas(proc_frame, bg='white', relief=tk.SUNKEN)
        self.proc_canvas.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Control buttons
        button_frame = ttk.Frame(self.conversion_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=10)

        ttk.Button(button_frame, text="Convert to SVG",
                   command=self.convert_to_svg).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Validate Quality",
                   command=self.validate_quality).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear",
                   command=self.clear_all).pack(side=tk.LEFT, padx=5)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self.conversion_frame, textvariable=self.status_var,
                  relief=tk.SUNKEN).grid(row=4, column=0, columnspan=2, sticky="ew", pady=5)

    def setup_settings_tab(self):
        # Benchmark section
        benchmark_frame = ttk.LabelFrame(self.settings_frame, text="Performance Benchmark", padding="10")
        benchmark_frame.grid(row=0, column=0, sticky="ew", pady=5)

        ttk.Button(benchmark_frame, text="Run Benchmark",
                   command=self.run_benchmark).pack(pady=10)

        self.benchmark_text = tk.Text(benchmark_frame, height=15, width=80)
        self.benchmark_text.pack(fill="both", expand=True)

        # Device info
        info_frame = ttk.LabelFrame(self.settings_frame, text="System Information", padding="10")
        info_frame.grid(row=1, column=0, sticky="ew", pady=5)

        device_info = f"PyTorch Device: {self.converter.device}\n"
        device_info += f"CUDA Available: {torch.cuda.is_available()}\n"
        if torch.cuda.is_available():
            device_info += f"GPU: {torch.cuda.get_device_name()}\n"
            device_info += f"CUDA Version: {torch.version.cuda}"

        ttk.Label(info_frame, text=device_info).pack(anchor="w")

    # ... (Implement the rest of the methods from the previous converter class)
    def browse_image(self):
        file_types = [
            ('Image files', '*.png *.jpg *.jpeg *.bmp *.tiff *.tif'),
            ('All files', '*.*')
        ]

        filename = filedialog.askopenfilename(filetypes=file_types)
        if filename:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, filename)
            self.load_image(filename)

    def load_image(self, filename):
        try:
            self.image_path = filename
            self.original_image = Image.open(filename).convert('RGB')
            self.update_previews()
            self.process_image()
            self.status_var.set(f"Loaded: {os.path.basename(filename)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")

    def on_size_change(self, value):
        size = int(float(value))
        self.size_label.config(text=f"{size} px")
        if self.original_image:
            self.process_image()

    def on_method_change(self):
        if self.original_image:
            self.process_image()

    def process_image(self):
        """Process image using selected method"""
        if not self.original_image:
            return

        try:
            method = self.conversion_method.get()
            size = self.output_size.get()

            # Convert to numpy for processing
            img_np = np.array(self.original_image.resize((size, size)))

            if method == "canny":
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                self.processed_image = cv2.Canny(gray, 50, 150)
            elif method == "adaptive":
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                self.processed_image = cv2.adaptiveThreshold(gray, 255,
                                                             cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                             cv2.THRESH_BINARY, 11, 2)
            elif method == "segmentation":
                self.processed_image = self.color_segmentation(img_np)
            elif method == "vector_quant":
                self.processed_image = self.vector_quantization(img_np)

            self.update_previews()

        except Exception as e:
            messagebox.showerror("Processing Error", f"Image processing failed: {str(e)}")

    def color_segmentation(self, image_np):
        """Color-based segmentation"""
        from sklearn.cluster import KMeans

        h, w = image_np.shape[:2]
        image_flat = image_np.reshape(-1, 3)

        kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
        labels = kmeans.fit_predict(image_flat)

        segmented = kmeans.cluster_centers_[labels].reshape(h, w, 3)
        return segmented.astype(np.uint8)

    def vector_quantization(self, image_np):
        """Vector quantization"""
        h, w = image_np.shape[:2]

        # Simple quantization by reducing color depth
        quantized = (image_np // 64) * 64
        return quantized

    def update_previews(self):
        """Update both preview canvases"""
        if self.original_image:
            # Original preview
            orig_preview = self.original_image.copy()
            self.update_canvas_preview(orig_preview, self.orig_canvas)

        if self.processed_image is not None:
            # Processed preview
            if len(self.processed_image.shape) == 2:
                proc_preview = Image.fromarray(self.processed_image, mode='L')
            else:
                proc_preview = Image.fromarray(self.processed_image, mode='RGB')
            self.update_canvas_preview(proc_preview, self.proc_canvas)

    def update_canvas_preview(self, image, canvas):
        """Update canvas with resized image"""
        canvas_size = 300
        image.thumbnail((canvas_size, canvas_size), Image.Resampling.LANCZOS)

        photo = ImageTk.PhotoImage(image)
        canvas.delete("all")
        canvas.config(width=image.width, height=image.height)
        canvas.create_image(image.width // 2, image.height // 2, image=photo, anchor=tk.CENTER)
        canvas.image = photo  # Keep reference

    def convert_to_svg(self):
        if not self.image_path or self.processed_image is None:
            messagebox.showwarning("Warning", "Please select and process an image first!")
            return

        try:
            output_path = filedialog.asksaveasfilename(
                defaultextension=".svg",
                filetypes=[('SVG files', '*.svg'), ('All files', '*.*')]
            )

            if output_path:
                if self.conversion_method.get() in ["canny", "adaptive"]:
                    svg_content = SVGGenerator.vector_svg_from_edges(
                        self.processed_image,
                        self.output_size.get(),
                        self.quality_setting.get()
                    )
                else:
                    if len(self.processed_image.shape) == 2:
                        proc_image = Image.fromarray(self.processed_image, mode='L').convert('RGB')
                    else:
                        proc_image = Image.fromarray(self.processed_image, mode='RGB')

                    svg_content = SVGGenerator.raster_svg_with_embedding(
                        proc_image,
                        self.output_size.get()
                    )

                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(svg_content)

                messagebox.showinfo("Success", f"SVG saved successfully!\n{output_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Conversion failed: {str(e)}")

    def validate_quality(self):
        """Validate conversion quality"""
        if not self.original_image or self.processed_image is None:
            messagebox.showwarning("Warning", "Please process an image first!")
            return

        try:
            original_np = np.array(self.original_image.resize(
                (self.output_size.get(), self.output_size.get())))

            report = QualityValidator.generate_validation_report(
                original_np,
                self.processed_image,
                self.conversion_method.get()
            )

            messagebox.showinfo("Quality Report",
                                f"Method: {report['method']}\n"
                                f"SSIM: {report['ssim']:.3f}\n"
                                f"Edge Preservation: {report['edge_preservation']:.3f}\n"
                                f"Overall Score: {report['overall_score']:.2f}/10")

        except Exception as e:
            messagebox.showerror("Validation Error", f"Quality validation failed: {str(e)}")

    def run_benchmark(self):
        """Run performance benchmark"""
        if not self.original_image:
            messagebox.showwarning("Warning", "Please load an image first!")
            return

        try:
            methods = ["canny", "segmentation", "adaptive", "vector_quant"]
            results = []

            for method in tqdm(methods, desc="Benchmarking"):
                self.conversion_method.set(method)
                start_time = time.time()
                self.process_image()
                end_time = time.time()

                # Validate quality
                original_np = np.array(self.original_image.resize(
                    (self.output_size.get(), self.output_size.get())))

                quality_report = QualityValidator.generate_validation_report(
                    original_np, self.processed_image, method)

                results.append({
                    'method': method,
                    'time': end_time - start_time,
                    'quality': quality_report['overall_score']
                })

            # Display results
            self.benchmark_text.delete(1.0, tk.END)
            self.benchmark_text.insert(tk.END, "Benchmark Results:\n\n")
            for result in results:
                self.benchmark_text.insert(tk.END,
                                           f"{result['method']:15} | Time: {result['time']:6.3f}s | "
                                           f"Quality: {result['quality']:5.2f}/10\n")

        except Exception as e:
            messagebox.showerror("Benchmark Error", f"Benchmark failed: {str(e)}")

    def clear_all(self):
        self.file_entry.delete(0, tk.END)
        self.image_path = None
        self.original_image = None
        self.processed_image = None
        self.orig_canvas.delete("all")
        self.proc_canvas.delete("all")
        self.status_var.set("Ready")