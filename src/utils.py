import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as ssim
import svgwrite
import io
import base64


class ImageUtils:
    """Utility functions for image processing"""

    @staticmethod
    def pil_to_cv2(pil_image):
        """Convert PIL Image to OpenCV format"""
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    @staticmethod
    def cv2_to_pil(cv2_image):
        """Convert OpenCV image to PIL format"""
        return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

    @staticmethod
    def resize_image(image, size, maintain_aspect=True):
        """Resize image with optional aspect ratio maintenance"""
        if maintain_aspect:
            image.thumbnail((size, size), Image.Resampling.LANCZOS)
        else:
            image = image.resize((size, size), Image.Resampling.LANCZOS)
        return image


class QualityValidator:
    """Class for validating conversion quality"""

    @staticmethod
    def calculate_ssim(original, processed):
        """Calculate Structural Similarity Index"""
        if len(processed.shape) == 2:
            processed = np.stack([processed] * 3, axis=-1)

        # Ensure same size
        min_size = min(original.shape[0], processed.shape[0])
        original_resized = cv2.resize(original, (min_size, min_size))
        processed_resized = cv2.resize(processed, (min_size, min_size))

        return ssim(original_resized, processed_resized,
                    multichannel=True, win_size=3,
                    data_range=processed_resized.max() - processed_resized.min())

    @staticmethod
    def calculate_edge_preservation(original, processed):
        """Calculate edge preservation metric"""
        original_edges = cv2.Canny(cv2.cvtColor(original, cv2.COLOR_RGB2GRAY), 50, 150)

        if len(processed.shape) == 2:
            processed_edges = cv2.Canny(processed, 50, 150)
        else:
            processed_edges = cv2.Canny(cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY), 50, 150)

        intersection = np.sum(original_edges & processed_edges)
        union = np.sum(original_edges | processed_edges)

        return intersection / union if union > 0 else 0

    @staticmethod
    def generate_validation_report(original, processed, method_name):
        """Generate comprehensive validation report"""
        ssim_score = QualityValidator.calculate_ssim(original, processed)
        edge_score = QualityValidator.calculate_edge_preservation(original, processed)

        # Calculate overall score
        overall_score = (ssim_score * 0.6 + edge_score * 0.4) * 10

        return {
            'ssim': ssim_score,
            'edge_preservation': edge_score,
            'overall_score': overall_score,
            'method': method_name
        }


class SVGGenerator:
    """Class for generating SVG files"""

    @staticmethod
    def vector_svg_from_edges(edges, output_size, quality='high'):
        """Create vector SVG from edge detection results"""
        # Threshold based on quality
        threshold_map = {'low': 0.3, 'medium': 0.5, 'high': 0.7, 'ultra': 0.8}
        threshold = threshold_map.get(quality, 0.7)

        _, binary = cv2.threshold(edges, threshold * 255, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(binary.astype(np.uint8),
                                       cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Create SVG
        dwg = svgwrite.Drawing(size=(output_size, output_size))

        # Add white background
        dwg.add(dwg.rect(insert=(0, 0),
                         size=('100%', '100%'),
                         fill='white'))

        # Add contours as paths
        for contour in contours:
            if len(contour) > 2:
                points = " ".join(f"{point[0][0]},{point[0][1]}" for point in contour)
                path_data = f"M {points} Z"
                dwg.add(dwg.path(d=path_data, fill='black', stroke='none'))

        return dwg.tostring()

    @staticmethod
    def raster_svg_with_embedding(image, output_size):
        """Create SVG with embedded raster image"""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG", optimize=True)
        img_str = buffered.getvalue()
        img_base64 = base64.b64encode(img_str).decode()

        svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{output_size}" height="{output_size}" xmlns="http://www.w3.org/2000/svg">
    <image width="{output_size}" height="{output_size}" 
           href="data:image/png;base64,{img_base64}"/>
</svg>'''

        return svg_content