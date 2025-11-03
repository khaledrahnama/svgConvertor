import unittest
import sys
import os
import torch
import numpy as np

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from PIL import Image
from models import EdgeDetectionModel, SVGPyTorchConverter


class TestPyTorchModels(unittest.TestCase):

    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_edge_detection_model_creation(self):
        """Test that edge detection model can be created"""
        model = EdgeDetectionModel()

        self.assertIsInstance(model, EdgeDetectionModel)
        self.assertTrue(hasattr(model, 'enc1'))
        self.assertTrue(hasattr(model, 'dec1'))
        self.assertTrue(hasattr(model, 'final'))

        # Test model parameters
        total_params = sum(p.numel() for p in model.parameters())
        self.assertGreater(total_params, 1000)  # Should have reasonable number of parameters

    def test_edge_detection_forward_pass(self):
        """Test model forward pass with dummy data"""
        model = EdgeDetectionModel().to(self.device)
        model.eval()

        # Create dummy input - note the model downsamples by 2, so output will be half size
        batch_size, channels, height, width = 1, 3, 64, 64
        expected_output_height = height // 2  # Due to maxpool layers
        expected_output_width = width // 2  # Due to maxpool layers

        dummy_input = torch.randn(batch_size, channels, height, width).to(self.device)

        with torch.no_grad():
            output = model(dummy_input)

        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (batch_size, 1, expected_output_height, expected_output_width))

        # Output should be between 0 and 1 due to sigmoid
        self.assertTrue(torch.all(output >= 0))
        self.assertTrue(torch.all(output <= 1))

        # Test with different input size
        dummy_input2 = torch.randn(2, 3, 128, 128).to(self.device)
        with torch.no_grad():
            output2 = model(dummy_input2)
        self.assertEqual(output2.shape, (2, 1, 64, 64))  # 128 -> 64 due to pooling

    def test_edge_detection_model_layers(self):
        """Test individual model layers"""
        model = EdgeDetectionModel()

        # Test encoder layers
        test_input = torch.randn(1, 3, 64, 64)

        # Test first encoder block
        e1 = model.enc1(test_input)
        self.assertEqual(e1.shape, (1, 64, 32, 32))  # Channels: 3->64, Size: 64->32

        # Test second encoder block
        e2 = model.enc2(e1)
        self.assertEqual(e2.shape, (1, 128, 16, 16))  # Channels: 64->128, Size: 32->16

    def test_converter_initialization(self):
        """Test SVG converter initialization"""
        converter = SVGPyTorchConverter()

        self.assertIsInstance(converter, SVGPyTorchConverter)
        self.assertTrue(hasattr(converter, 'device'))
        self.assertTrue(hasattr(converter, 'edge_model'))
        self.assertIsInstance(converter.edge_model, EdgeDetectionModel)

    def test_converter_preprocessing(self):
        """Test image preprocessing"""
        converter = SVGPyTorchConverter()

        # Create test image
        test_image = Image.new('RGB', (100, 100), color='blue')
        processed = converter.preprocess_image(test_image, 64)

        self.assertIsInstance(processed, torch.Tensor)
        self.assertEqual(processed.shape[0], 1)  # batch size
        self.assertEqual(processed.shape[1], 3)  # channels
        self.assertEqual(processed.shape[2], 64)  # height
        self.assertEqual(processed.shape[3], 64)  # width

        # Test tensor values are normalized
        self.assertLess(torch.max(processed), 3.0)  # After normalization
        self.assertGreater(torch.min(processed), -3.0)

    def test_converter_different_sizes(self):
        """Test converter with different image sizes"""
        converter = SVGPyTorchConverter()

        sizes = [32, 64, 128]
        for size in sizes:
            test_image = Image.new('RGB', (100, 100), color='red')
            processed = converter.preprocess_image(test_image, size)
            self.assertEqual(processed.shape[2], size)  # height
            self.assertEqual(processed.shape[3], size)  # width

    def test_model_on_actual_image(self):
        """Test model with actual image data"""
        model = EdgeDetectionModel().to(self.device)
        model.eval()

        # Create a more realistic test image (gradient)
        width, height = 64, 64
        test_image = Image.new('RGB', (width, height))
        pixels = []
        for y in range(height):
            for x in range(width):
                r = int(255 * x / width)
                g = int(255 * y / height)
                b = 128
                pixels.append((r, g, b))
        test_image.putdata(pixels)

        converter = SVGPyTorchConverter()
        processed = converter.preprocess_image(test_image, 64)

        with torch.no_grad():
            output = model(processed)

        self.assertEqual(output.shape, (1, 1, 32, 32))
        self.assertTrue(torch.all(output >= 0))
        self.assertTrue(torch.all(output <= 1))

        # Output should have some variation (not all zeros or ones)
        output_std = torch.std(output)
        self.assertGreater(output_std, 0.01)


class TestModelRobustness(unittest.TestCase):
    """Test model robustness with edge cases"""

    def test_model_extreme_inputs(self):
        """Test model with extreme input values"""
        model = EdgeDetectionModel()
        model.eval()

        # Test with all zeros
        zeros_input = torch.zeros(1, 3, 64, 64)
        with torch.no_grad():
            zeros_output = model(zeros_input)

        # Test with all ones
        ones_input = torch.ones(1, 3, 64, 64)
        with torch.no_grad():
            ones_output = model(ones_input)

        # Both should produce valid outputs
        self.assertTrue(torch.all(zeros_output >= 0))
        self.assertTrue(torch.all(zeros_output <= 1))
        self.assertTrue(torch.all(ones_output >= 0))
        self.assertTrue(torch.all(ones_output <= 1))

    def test_model_batch_processing(self):
        """Test model with batch inputs"""
        model = EdgeDetectionModel()
        model.eval()

        batch_sizes = [1, 2, 4]
        for batch_size in batch_sizes:
            batch_input = torch.randn(batch_size, 3, 64, 64)
            with torch.no_grad():
                batch_output = model(batch_input)

            self.assertEqual(batch_output.shape[0], batch_size)
            self.assertEqual(batch_output.shape[1], 1)  # Single channel output
            self.assertEqual(batch_output.shape[2], 32)  # Height after pooling
            self.assertEqual(batch_output.shape[3], 32)  # Width after pooling


def suite():
    """Create test suite"""
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(TestPyTorchModels))
    test_suite.addTest(unittest.makeSuite(TestModelRobustness))
    return test_suite


if __name__ == '__main__':
    # Run specific tests
    runner = unittest.TextTestRunner(verbosity=2)

    # Run all tests
    unittest.main(verbosity=2)