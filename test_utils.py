import unittest
from utils import validate_phone_number, format_shopping_list, categorize_item
from image_utils import preprocess_image, has_sufficient_quality
from PIL import Image
import numpy as np
import cv2
import os

class TestUtils(unittest.TestCase):
    def test_phone_number_validation(self):
        """Test phone number validation function"""
        valid_numbers = [
            "+12345678901",
            "+442012345678",
            "+6512345678"
        ]
        invalid_numbers = [
            "12345678901",  # missing +
            "+1234",  # too short
            "abcdefghijk",  # non-numeric
            "+12345abcdef"  # mixed characters
        ]
        
        for number in valid_numbers:
            self.assertTrue(validate_phone_number(number))
            
        for number in invalid_numbers:
            self.assertFalse(validate_phone_number(number))
            
    def test_shopping_list_formatting(self):
        """Test shopping list formatting function"""
        items = [
            {"name": "apple", "quantity": 3, "to_purchase": True},
            {"name": "banana", "quantity": 2, "to_purchase": False},
            {"name": "milk", "quantity": 1, "to_purchase": True}
        ]
        
        formatted = format_shopping_list(items)
        expected = "Shopping List:\n• 3 apple\n• 1 milk\n"
        self.assertEqual(formatted, expected)
        
    def test_food_categorization(self):
        """Test food categorization function"""
        test_cases = [
            ("apple", "fruits"),
            ("carrot", "vegetables"),
            ("chicken", "meat"),
            ("milk", "dairy"),
            ("unknown_item", "other")
        ]
        
        for item, expected_category in test_cases:
            self.assertEqual(categorize_item(item), expected_category)

class TestImageProcessing(unittest.TestCase):
    def setUp(self):
        """Create test images for processing tests"""
        # Create a directory for test images if it doesn't exist
        self.test_dir = os.path.join(os.path.dirname(__file__), 'test_images')
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create a good quality test image
        self.good_image = Image.fromarray(np.random.randint(100, 200, (300, 400, 3), dtype=np.uint8))
        self.good_image_path = os.path.join(self.test_dir, 'good_quality.jpg')
        self.good_image.save(self.good_image_path)
        
        # Create a poor quality (dark) test image
        self.dark_image = Image.fromarray(np.random.randint(0, 50, (300, 400, 3), dtype=np.uint8))
        self.dark_image_path = os.path.join(self.test_dir, 'dark_quality.jpg')
        self.dark_image.save(self.dark_image_path)
        
        # Create a low resolution test image
        self.low_res_image = Image.fromarray(np.random.randint(100, 200, (100, 150, 3), dtype=np.uint8))
        self.low_res_path = os.path.join(self.test_dir, 'low_res.jpg')
        self.low_res_image.save(self.low_res_path)

    def tearDown(self):
        """Clean up test images after tests"""
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_image_preprocessing(self):
        """Test image preprocessing functionality"""
        # Test preprocessing maintains image dimensions
        processed = preprocess_image(self.good_image)
        self.assertEqual(processed.size, self.good_image.size)
        
        # Test preprocessing handles grayscale images
        gray_image = self.good_image.convert('L')
        processed_gray = preprocess_image(gray_image)
        self.assertEqual(len(np.array(processed_gray).shape), 3)  # Should convert to RGB
        
        # Test preprocessing handles RGBA images
        rgba_image = self.good_image.convert('RGBA')
        processed_rgba = preprocess_image(rgba_image)
        self.assertEqual(len(np.array(processed_rgba).shape), 3)  # Should convert to RGB

    def test_quality_validation(self):
        """Test image quality validation functionality"""
        # Good quality image should pass
        self.assertTrue(has_sufficient_quality(self.good_image))
        
        # Dark image should fail
        self.assertFalse(has_sufficient_quality(self.dark_image))
        
        # Low resolution image should fail
        self.assertFalse(has_sufficient_quality(self.low_res_image))

class TestFoodDetection(unittest.TestCase):
    def setUp(self):
        from utils import FoodDetector
        self.detector = FoodDetector()

    def test_detection_parsing(self):
        """Test VLM response parsing"""
        sample_response = """1 apple
2 bananas
500g chicken
1 loaf bread"""
        items = self.detector._parse_vlm_response(sample_response)
        
        # Check the number of detected items
        self.assertEqual(len(items), 4)
        
        # Check specific items are parsed correctly
        apple_item = next(item for item in items if item["name"] == "apple")
        self.assertEqual(apple_item["quantity_est"], "1")
        
        banana_item = next(item for item in items if item["name"] == "banana")
        self.assertEqual(banana_item["quantity_est"], "2")

    def test_merge_detections(self):
        """Test merging multiple detections"""
        detections1 = [
            {"name": "apple", "quantity_est": "2", "confidence": 0.8},
            {"name": "banana", "quantity_est": "3", "confidence": 0.9}
        ]
        
        detections2 = [
            {"name": "apple", "quantity_est": "3", "confidence": 0.7},
            {"name": "orange", "quantity_est": "1", "confidence": 0.85}
        ]
        
        # Merge detections
        all_detections = detections1 + detections2
        from app import ImageProcessingAgent
        agent = ImageProcessingAgent()
        merged = agent._merge_detections(all_detections)
        
        # Check that duplicates are properly merged
        self.assertEqual(len(merged), 3)  # Should have apple, banana, orange
        
        # Check that quantities are averaged and max confidence is used
        apple_item = next(item for item in merged if item["name"] == "apple")
        self.assertEqual(float(apple_item["quantity_est"]), 2.5)  # Average of 2 and 3
        self.assertEqual(apple_item["confidence"], 0.8)  # Max confidence

    def test_basic_image_processing_fallback(self):
        """Test the basic image processing fallback method"""
        # Create a test image
        test_image = os.path.join(os.path.dirname(__file__), 'test_images', 'test.jpg')
        img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        cv2.imwrite(test_image, img)
        
        # Test fallback detection
        items = self.detector._basic_image_processing(test_image)
        
        # Check that we get some results
        self.assertTrue(len(items) > 0)
        
        # Check item structure
        for item in items:
            self.assertIn("name", item)
            self.assertIn("quantity_est", item)
            self.assertTrue(float(item["quantity_est"]) >= 1)

if __name__ == '__main__':
    unittest.main()
