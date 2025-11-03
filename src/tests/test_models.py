import unittest
import sys
import os
import torch
import numpy as np

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from PIL import Image

# Import models after adding to path
from models import EdgeDetectionModel, SVGPyTorchConverter


class TestPyTorchModels(unittest.TestCase):

    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Testing on device: {self.device}")

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
        print(f"Model has {total_params} parameters")

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
        expected_shape = (batch_size, 1, expected_output_height, expected_output_width)
        self.assertEqual(output.shape, expected_shape)

        # Output should be between 0 and 1 due to sigmoid
        self.assertTrue(torch.all(output >= 0))
        self.assertTrue(torch.all(output <= 1))

        print(f"Input shape: {dummy_input.shape}, Output shape: {output.shape}")

    def test_edge_detection_model_layers(self):
        """Test individual model layers"""
        model = EdgeDetectionModel()
        model.eval()

        # Test encoder layers
        test_input = torch.randn(1, 3, 64, 64)

        with torch.no_grad():
            # Test first encoder block
            e1 = model.enc1(test_input)
            self.assertEqual(e1.shape, (1, 64, 32, 32))  # Channels: 3->64, Size: 64->32

            # Test second encoder block
            e2 = model.enc2(e1)
            self.assertEqual(e2.shape, (1, 128, 16, 16))  # Channels: 64->128, Size: 32->16

        print("Layer dimension tests passed")

    def test_converter_initialization(self):
        """Test SVG converter initialization"""
        converter = SVGPyTorchConverter()

        self.assertIsInstance(converter, SVGPyTorchConverter)
        self.assertTrue(hasattr(converter, 'device'))
        self.assertTrue(hasattr(converter, 'edge_model'))
        self.assertIsInstance(converter.edge_model, EdgeDetectionModel)

        print("Converter initialization test passed")

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

        print("Image preprocessing test passed")

    def test_converter_different_sizes(self):
        """Test converter with different image sizes"""
        converter = SVGPyTorchConverter()

        sizes = [32, 64, 128]
        for size in sizes:
            test_image = Image.new('RGB', (100, 100), color='red')
            processed = converter.preprocess_image(test_image, size)
            self.assertEqual(processed.shape[2], size)  # height
            self.assertEqual(processed.shape[3], size)  # width

        print("Multiple size preprocessing test passed")

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

        print("Actual image processing test passed")


class TestModelRobustness(unittest.TestCase):
    """Test model robustness with edge cases"""

    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

        print("Extreme input tests passed")

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

        print("Batch processing tests passed")


def run_tests():
    """Run all tests and return results"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestPyTorchModels)
    suite.addTests(loader.loadTestsFromTestCase(TestModelRobustness))

    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


#something
if __name__ == '__main__':
    print("Running PyTorch Models Tests...")
    print("=" * 50)

    result = run_tests()

    print("=" * 50)
    if result.wasSuccessful():
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")

    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)