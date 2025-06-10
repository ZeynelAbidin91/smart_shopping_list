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

# Set environment variables for better CUDA performance
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Enable for debugging if needed

from PIL import Image
from typing import Dict, List, Any, Optional
from huggingface_hub import InferenceClient
from llama_index.core.tools import BaseTool
from loguru import logger
from pathlib import Path
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor
)

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*Video processor config.*')
warnings.filterwarnings('ignore', message='.*configure the model properly.*')

class ImageAnalysisTool(BaseTool):
    """Tool for analyzing images using Qwen2.5 VLM for food detection."""
    name = "image_analyzer"
    description = "Analyzes images to detect food items and their quantities"

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return metadata for the tool."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the image file to analyze"
                }
            }
        }

    def __init__(self):
        super().__init__()
        self.model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
        
        # Try to use CUDA first, fallback to CPU if issues occur
        try:
            if torch.cuda.is_available():
                self.device = "cuda"
                # Test CUDA functionality
                test_tensor = torch.randn(1, device="cuda")
                del test_tensor
                torch.cuda.empty_cache()
                print(f"Using device: {self.device}")
                print(f"GPU: {torch.cuda.get_device_name()}")
                print(f"CUDA Version: {torch.version.cuda}")
                print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            else:
                self.device = "cpu"
                print("CUDA not available, using CPU")
        except Exception as e:
            print(f"CUDA test failed: {str(e)}")
            self.device = "cpu"
            print("Falling back to CPU")
            
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                use_fast=True  # Enable fast tokenizer
            )
            
            # Configure model loading based on device
            model_kwargs = {
                "low_cpu_mem_usage": True
            }
            
            if self.device == "cuda":
                model_kwargs.update({
                    "torch_dtype": torch.float16,
                    "device_map": "auto",
                    "max_memory": {0: "7GiB"}  # Leave some memory for other processes
                })
            else:
                model_kwargs.update({
                    "torch_dtype": torch.float32
                })
            
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Move to CPU if not using device_map
            if self.device == "cpu":
                self.model = self.model.to(self.device)
                
            print(f"Model successfully loaded on {self.device}")
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            print(f"Model loading failed, falling back to CPU if not already")
            
            # Try loading on CPU as fallback
            if self.device != "cpu":
                self.device = "cpu"
                try:
                    self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=True
                    ).to(self.device)
                    self.processor = AutoProcessor.from_pretrained(self.model_name, use_fast=True)
                    print("Successfully loaded model on CPU as fallback")
                except Exception as e2:
                    logger.error(f"CPU fallback also failed: {str(e2)}")
                    self.model = None
                    self.processor = None
            else:
                self.model = None
                self.processor = None

    def _format_prompt(self, image) -> str:
        """Format the prompt for food detection."""
        return "Look at this image and list all food items you can see. For each item, provide the quantity (if visible) and name. Format your response as a comma-separated list like '2 apples, 1 milk carton, 3 eggs'. Focus on food items that might need restocking."
        
    def __call__(self, image_path: str) -> List[Dict[str, Any]]:
        """Process image and detect food items."""
        if not self.model or not self.processor:
            return self._fallback_detection()
        
        try:
            # Load and process image
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Prepare prompt - use proper format for Qwen2.5-VL
            prompt = self._format_prompt(image)
            
            # Create conversation format expected by Qwen2.5-VL
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Apply chat template
            text_input = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Process inputs
            inputs = self.processor(
                text=[text_input], 
                images=[image], 
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                    temperature=None,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode response
            input_len = inputs['input_ids'].shape[1]
            response = self.processor.decode(
                outputs[0][input_len:], 
                skip_special_tokens=True
            )
            
            return self._parse_model_response(response)
            
        except Exception as e:
            logger.error(f"Error in image analysis: {str(e)}")
            return self._fallback_detection()

    def _parse_model_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse the model's response into structured format."""
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

    def _fallback_detection(self) -> List[Dict[str, Any]]:
        """Fallback method when image processing fails."""
        return [{"name": "unknown item", "quantity": "1"}]

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
