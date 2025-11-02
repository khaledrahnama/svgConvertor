import unittest
import tempfile
import os
import sys
import numpy as np

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from PIL import Image
import cv2

# Import from your modules
from converter import SVGPyTorchGUI
from utils import QualityValidator, SVGGenerator, ImageUtils


class TestSVGConverter(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures before each test method"""
        # Create a test image
        self.test_image = Image.new('RGB', (100, 100), color='red')
        self.test_image_np = np.array(self.test_image)

    def test_quality_validation_ssim(self):
        """Test SSIM calculation"""
        # Create a similar but different image
        processed = np.ones((100, 100, 3), dtype=np.uint8) * 200

        ssim_score = QualityValidator.calculate_ssim(self.test_image_np, processed)

        self.assertIsInstance(ssim_score, float)
        self.assertGreaterEqual(ssim_score, 0.0)
        self.assertLessEqual(ssim_score, 1.0)

    def test_quality_validation_edge_preservation(self):
        """Test edge preservation calculation"""
        # Create edge images
        original_edges = cv2.Canny(cv2.cvtColor(self.test_image_np, cv2.COLOR_RGB2GRAY), 50, 150)
        processed_edges = cv2.Canny(cv2.cvtColor(self.test_image_np, cv2.COLOR_RGB2GRAY), 50, 150)

        edge_score = QualityValidator.calculate_edge_preservation(self.test_image_np, processed_edges)

        self.assertIsInstance(edge_score, float)
        self.assertGreaterEqual(edge_score, 0.0)
        self.assertLessEqual(edge_score, 1.0)

    def test_quality_validation_report(self):
        """Test comprehensive validation report generation"""
        processed = np.ones((100, 100, 3), dtype=np.uint8) * 128

        report = QualityValidator.generate_validation_report(
            self.test_image_np, processed, "test_method"
        )

        self.assertIsInstance(report, dict)
        self.assertIn('ssim', report)
        self.assertIn('edge_preservation', report)
        self.assertIn('overall_score', report)
        self.assertIn('method', report)
        self.assertEqual(report['method'], 'test_method')

    def test_svg_generation_raster(self):
        """Test raster SVG generation"""
        svg_content = SVGGenerator.raster_svg_with_embedding(self.test_image, 100)

        self.assertIsInstance(svg_content, str)
        self.assertIn('svg', svg_content.lower())
        self.assertIn('100', svg_content)
        self.assertIn('image', svg_content.lower())

    def test_svg_generation_vector(self):
        """Test vector SVG generation from edges"""
        # Create a simple edge image
        edges = np.zeros((100, 100), dtype=np.uint8)
        edges[40:60, 40:60] = 255  # Create a square

        svg_content = SVGGenerator.vector_svg_from_edges(edges, 100, 'high')

        self.assertIsInstance(svg_content, str)
        self.assertIn('svg', svg_content.lower())
        self.assertIn('path', svg_content.lower())

    def test_image_utils_conversions(self):
        """Test image format conversion utilities"""
        # Test PIL to OpenCV conversion
        cv2_image = ImageUtils.pil_to_cv2(self.test_image)
        self.assertIsInstance(cv2_image, np.ndarray)
        self.assertEqual(cv2_image.shape, (100, 100, 3))

        # Test OpenCV to PIL conversion
        pil_image = ImageUtils.cv2_to_pil(cv2_image)
        self.assertIsInstance(pil_image, Image.Image)
        self.assertEqual(pil_image.size, (100, 100))

    def test_image_resize(self):
        """Test image resizing utility"""
        resized_image = ImageUtils.resize_image(self.test_image, 50, maintain_aspect=True)

        self.assertIsInstance(resized_image, Image.Image)
        # Should maintain aspect ratio
        self.assertLessEqual(resized_image.size[0], 50)
        self.assertLessEqual(resized_image.size[1], 50)


class TestImageProcessingMethods(unittest.TestCase):

    def setUp(self):
        self.test_image = Image.new('RGB', (100, 100), color=(100, 150, 200))
        self.test_image_np = np.array(self.test_image)

    def test_canny_edge_detection(self):
        """Test Canny edge detection produces valid output"""
        gray = cv2.cvtColor(self.test_image_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        self.assertIsInstance(edges, np.ndarray)
        self.assertEqual(edges.shape, (100, 100))
        # Should be binary image (0 or 255)
        self.assertTrue(np.all(np.isin(edges, [0, 255])))

    def test_adaptive_threshold(self):
        """Test adaptive thresholding"""
        gray = cv2.cvtColor(self.test_image_np, cv2.COLOR_RGB2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

        self.assertIsInstance(binary, np.ndarray)
        self.assertEqual(binary.shape, (100, 100))
        # Should be binary image
        self.assertTrue(np.all(np.isin(binary, [0, 255])))


def suite():
    """Create test suite"""
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(TestSVGConverter))
    test_suite.addTest(unittest.makeSuite(TestImageProcessingMethods))
    return test_suite


if __name__ == '__main__':
    # Run specific tests
    runner = unittest.TextTestRunner(verbosity=2)

    # Run all tests
    unittest.main(verbosity=2)