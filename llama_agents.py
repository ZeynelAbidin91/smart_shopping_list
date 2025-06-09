"""
LlamaIndex-based agents for the Smart Fridge Shopping List Generator.
"""

import os
import re
import uuid
import warnings
import torch
import json
import qrcode
import io
import base64
from PIL import Image
from typing import Dict, List, Any, Optional
from huggingface_hub import InferenceClient
from llama_index.core.tools import BaseTool
from loguru import logger
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*Video processor config.*')
warnings.filterwarnings('ignore', message='.*configure the model properly.*')

class ImageAnalysisTool(BaseTool):
    """Tool for analyzing images using Qwen2.5 VLM through Hugging Face Inference API."""
    name = "image_analyzer"
    description = "Analyzes images to detect food items and their quantities"
    api_base = "https://api-inference.huggingface.co/models"
    
    def __init__(self):
        super().__init__()
        self.model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
        self.client = None
        
        try:
            # Use Hugging Face API token if available
            hf_token = os.getenv("HUGGINGFACE_TOKEN")
            if not hf_token:
                logger.warning("HUGGINGFACE_TOKEN not found in environment variables")
                logger.info("Falling back to default detection")
                return
            
            # Initialize the inference client
            self.client = InferenceClient(token=hf_token)
            logger.info("Successfully initialized Hugging Face Inference API")
        except Exception as e:
            logger.error(f"Error initializing Hugging Face API: {str(e)}")
            logger.info("Falling back to default detection")

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get metadata about the tool."""
        return {
            "name": self.name,
            "description": self.description,
            "model_name": self.model_name
        }    def _format_input(self, image_path: str) -> bytes:
        """Format the input image for the model API."""
        with Image.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
            return buffer.getvalue()

    def __call__(self, image_path: str) -> List[Dict[str, Any]]:
        """Process image and detect food items."""
        if not self.client:
            return self._fallback_detection()
            
        try:
            # Format prompt and send request to Hugging Face
            inputs = self._format_prompt(image_path)
            response = self.client.post(
                json=inputs,
                model=self.model_name
            )
            
            # Parse the response
            return self._parse_model_response(response)
            
        except Exception as e:
            logger.error(f"Error in image analysis: {str(e)}")
            return self._fallback_detection()

    def _parse_model_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse the model's response into structured format."""
        try:
            items = []
            # Split on commas and process each item
            for item in response.split(","):
                item = item.strip()
                if not item:
                    continue
                    
                # Try to extract quantity and name
                match = re.search(r"(\d+(?:\.\d+)?(?:\s*[a-zA-Z]+)?)\s*(.*)", item)
                if match:
                    quantity, name = match.groups()
                else:
                    parts = item.split()
                    if len(parts) > 1 and parts[0].replace(".", "").isdigit():
                        quantity = parts[0]
                        name = " ".join(parts[1:])
                    else:
                        quantity = "1"
                        name = item
                        
                items.append({
                    "name": name.strip().lower(),
                    "quantity": quantity.strip()
                })
            return items
        except Exception as e:
            logger.error(f"Error parsing model response: {str(e)}")
            return []

    def _fallback_detection(self) -> List[Dict[str, Any]]:
        """Fallback method when image processing fails."""
        return [
            {"name": "unknown item", "quantity": "1"}
        ]

class ShoppingListItem:
    """Represents a single item in the shopping list."""
    def __init__(self, name: str, quantity: str):
        self.name = name.lower().strip()
        self.quantity = self._normalize_quantity(quantity)
        
    def _normalize_quantity(self, quantity: str) -> str:
        """Normalize quantity to a standard format."""
        try:
            # Remove any trailing units for now
            quantity = re.match(r"[\d.]+", quantity)[0]
            return str(float(quantity))
        except:
            return "1"
    
    def __eq__(self, other):
        return isinstance(other, ShoppingListItem) and self.name == other.name
    
    def __hash__(self):
        return hash(self.name)

class ShoppingListAgent:
    """Agent for managing shopping list generation from image analysis."""
    def __init__(self):
        self.image_analyzer = ImageAnalysisTool()
        self.items = set()  # Use set to avoid duplicates
        self.qr_output_dir = Path("qr_codes")
        self.qr_output_dir.mkdir(exist_ok=True)
        
    def add_from_image(self, image_path: str) -> bool:
        """Analyze an image and add detected items to the shopping list."""
        try:
            detected_items = self.image_analyzer(image_path)
            for item in detected_items:
                self.items.add(ShoppingListItem(item["name"], item["quantity"]))
            return True
        except Exception as e:
            logger.error(f"Failed to process image {image_path}: {str(e)}")
            return False
    
    def clear_list(self):
        """Clear the current shopping list."""
        self.items.clear()
    
    def get_list(self) -> List[Dict[str, str]]:
        """Get the current shopping list in a structured format."""
        return [{"name": item.name, "quantity": item.quantity} 
                for item in sorted(self.items, key=lambda x: x.name)]
    
    def merge_quantities(self, items: List[ShoppingListItem]) -> List[ShoppingListItem]:
        """Merge quantities for duplicate items."""
        merged = {}
        for item in items:
            if item.name in merged:
                try:
                    merged[item.name].quantity = str(
                        float(merged[item.name].quantity) + float(item.quantity)
                    )
                except ValueError:
                    # If conversion fails, keep the larger quantity
                    if float(item.quantity) > float(merged[item.name].quantity):
                        merged[item.name].quantity = item.quantity
            else:
                merged[item.name] = item
        return list(merged.values())
    
    def generate_qr_code(self, filename: str = None) -> str:
        """Generate QR code for the current shopping list.
        
        Args:
            filename (str, optional): Name for the QR code file. 
                If None, a UUID will be generated.
                
        Returns:
            str: Path to the generated QR code image
        """
        try:
            # Get current list and convert to JSON
            shopping_list = self.get_list()
            list_data = json.dumps(shopping_list)
            
            # Generate QR code
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=4,
            )
            qr.add_data(list_data)
            qr.make(fit=True)
            
            # Create QR code image
            qr_image = qr.make_image(fill_color="black", back_color="white")
            
            # Generate filename if not provided
            if not filename:
                filename = f"shopping_list_{uuid.uuid4().hex[:8]}.png"
            
            # Ensure .png extension
            if not filename.lower().endswith('.png'):
                filename += '.png'
                
            # Save QR code
            output_path = self.qr_output_dir / filename
            qr_image.save(output_path)
            logger.info(f"Generated QR code: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to generate QR code: {str(e)}")
            return ""
    
    def decode_qr_code(self, image_path: str) -> bool:
        """Decode a shopping list from a QR code image.
        
        Args:
            image_path (str): Path to the QR code image
            
        Returns:
            bool: True if successfully decoded and added items
        """
        try:
            # Open image
            image = Image.open(image_path)
            
            # Decode QR code
            qr = qrcode.QRCode()
            qr.add_data("")  # Clear any existing data
            qr.decode(image)
            data = qr.get_matrix()
            
            if not data:
                logger.error("No QR code data found in image")
                return False
                
            # Parse JSON data
            shopping_list = json.loads(data)
            
            # Add items to current list
            for item in shopping_list:
                self.items.add(ShoppingListItem(item["name"], item["quantity"]))
                
            logger.info(f"Successfully imported {len(shopping_list)} items from QR code")
            return True
            
        except Exception as e:
            logger.error(f"Failed to decode QR code: {str(e)}")
            return False
    
    def process_image(self, image_path: str, preferences: Dict[str, Any] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Process an image and generate a shopping list with consideration for preferences.
        
        Args:
            image_path (str): Path to the image file
            preferences (Dict[str, Any], optional): User preferences including diet and cuisines
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: Dictionary with detected items and final filtered list
        """
        try:
            # Clear existing list
            self.clear_list()
            
            # Detect items from image
            detected_items = self.image_analyzer(image_path)
            
            # Add detected items to list
            for item in detected_items:
                self.items.add(ShoppingListItem(item["name"], item["quantity"]))
            
            # Get the final list
            final_list = self.get_list()
            
            return {
                "detected_items": detected_items,
                "final_list": final_list
            }
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return {
                "detected_items": [],
                "final_list": []
            }
