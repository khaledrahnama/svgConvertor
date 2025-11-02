import unittest
import sys
import os
import torch
import numpy as np

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models import EdgeDetectionModel, SVGPyTorchConverter
from PIL import Image


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

    def test_edge_detection_forward_pass(self):
        """Test model forward pass with dummy data"""
        model = EdgeDetectionModel().to(self.device)
        model.eval()

        # Create dummy input
        batch_size, channels, height, width = 1, 3, 64, 64
        dummy_input = torch.randn(batch_size, channels, height, width).to(self.device)

        with torch.no_grad():
            output = model(dummy_input)

        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (batch_size, 1, height, width))
        # Output should be between 0 and 1 due to sigmoid
        self.assertTrue(torch.all(output >= 0))
        self.assertTrue(torch.all(output <= 1))

    def test_converter_initialization(self):
        """Test SVG converter initialization"""
        converter = SVGPyTorchConverter()

        self.assertIsInstance(converter, SVGPyTorchConverter)
        self.assertTrue(hasattr(converter, 'device'))
        self.assertTrue(hasattr(converter, 'edge_model'))

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


if __name__ == '__main__':
    unittest.main(verbosity=2)