"""
LlamaIndex-based agents for the Smart Fridge Shopping List Generator.
"""

import os
import re
import warnings
import torch
import json

# Set OpenMP environment variable to handle multiple runtime versions
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Set environment variables for better CUDA performance
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Enable for better debugging

from PIL import Image
from typing import Dict, List, Any, Optional
from llama_index.core.tools import BaseTool
from loguru import logger
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
        self.device = "cpu"  # Default to CPU
        self.model = None
        self.processor = None
        
        # Initialize device with safer approach
        self._initialize_device()
        
        # Load model and processor
        self._load_model()

    def _initialize_device(self):
        """Initialize device with safer CUDA testing."""
        if not torch.cuda.is_available():
            print("CUDA not available, using CPU")
            return
            
        try:
            print("CUDA is available, testing compatibility...")
            
            # Basic CUDA test - less likely to trigger kernel errors
            with torch.cuda.device(0):
                # Simple tensor creation and operations
                test_tensor = torch.ones(10, device="cuda")
                result = test_tensor + 1
                
                # If we get here, basic CUDA works
                device_info = torch.cuda.get_device_properties(0)
                print(f"✓ GPU: {torch.cuda.get_device_name()}")
                print(f"✓ CUDA Capability: {device_info.major}.{device_info.minor}")
                print(f"✓ GPU Memory: {device_info.total_memory / 1024**3:.1f} GB")
                
                # Don't immediately switch to CUDA - we'll handle device selection in model loading
                # based on model compatibility with this GPU
                
                # Clean up test tensors
                del test_tensor, result
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"✗ CUDA test failed: {str(e)}")
            print("✗ Falling back to CPU mode for compatibility")
            # Clean up any GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _load_model(self):
        """Load model with safe fallback strategies."""
        try:
            print("Loading model processor...")
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                use_fast=True
            )
            print("✓ Processor loaded successfully")
            
            # First try: Load with automatic device mapping (safest approach)
            print(f"Loading {self.model_name} model with automatic device mapping...")
            
            # Configure model loading - use lower precision for CUDA
            model_kwargs = {
                "low_cpu_mem_usage": True,
                "torch_dtype": torch.float16,
                "device_map": "auto"
            }
            
            # Try loading with auto device mapping
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Check where model was loaded
            if any(p.device.type == "cuda" for p in self.model.parameters()):
                self.device = "cuda"
                print(f"✓ Model loaded successfully with GPU acceleration")
            else:
                self.device = "cpu"
                print(f"✓ Model loaded on CPU (automatic device mapping decision)")
                
        except Exception as e:
            print(f"✗ Automatic device mapping failed: {str(e)}")
            
            # Second try: Explicit CPU loading
            try:
                print("Trying explicit CPU loading...")
                self.device = "cpu"
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                ).to("cpu")
                print("✓ Successfully loaded model on CPU")
            except Exception as e2:
                print(f"✗ CPU loading also failed: {str(e2)}")
                self.model = None
                logger.error("Complete model loading failure - using fallback detection")    
    def _format_prompt(self, image) -> str:
        """Format the prompt for food detection."""
        return "Look at this image and list all food items you can see. For each item, provide the quantity (if visible) and name. Format your response as a comma-separated list like '2 apples, 1 milk carton, 3 eggs'. Focus on food items that might need restocking."
        
    def __call__(self, image_path: str) -> List[Dict[str, Any]]:
        """Process image and detect food items."""
        if not self.model or not self.processor:
            logger.warning("Model or processor not available, using fallback detection")
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
            
            # Process inputs - first try with current device (could be GPU)
            try:
                logger.info(f"Processing inputs using device: {self.device}")
                inputs = self.processor(
                    text=[text_input], 
                    images=[image], 
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                
                # Generate response with safer settings
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=False,
                        temperature=None,
                        pad_token_id=self.processor.tokenizer.eos_token_id
                    )
                
            except RuntimeError as cuda_err:
                # Check if this is a CUDA error
                if "CUDA" in str(cuda_err) or "cuda" in str(cuda_err) or "kernel image" in str(cuda_err):
                    logger.warning(f"CUDA error during image processing: {str(cuda_err)}")
                    logger.warning("Falling back to CPU mode for this operation")
                    
                    # If we were using CUDA, switch to CPU for this operation
                    if self.device == "cuda":
                        # Process inputs on CPU instead
                        inputs = self.processor(
                            text=[text_input], 
                            images=[image], 
                            return_tensors="pt",
                            padding=True
                        ).to("cpu")
                        
                        # Move model to CPU temporarily for this operation
                        logger.info("Moving model to CPU for this operation")
                        self.model = self.model.to("cpu")
                        
                        # Generate response on CPU
                        with torch.no_grad():
                            outputs = self.model.generate(
                                **inputs,
                                max_new_tokens=100,
                                do_sample=False,
                                temperature=None,
                                pad_token_id=self.processor.tokenizer.eos_token_id
                            )
                    else:
                        # If we're already on CPU and still getting CUDA errors, something else is wrong
                        raise cuda_err
                else:
                    # Not a CUDA error, re-raise
                    raise cuda_err
            
            # Decode response
            input_len = inputs['input_ids'].shape[1]
            response = self.processor.decode(
                outputs[0][input_len:], 
                skip_special_tokens=True
            )
            
            # Clean up CUDA memory if we're using it
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return self._parse_model_response(response)
            
        except Exception as e:
            logger.error(f"Error in image analysis: {str(e)}")
            # If we get a CUDA error, permanently switch to CPU mode
            if "CUDA" in str(e) or "cuda" in str(e) or "kernel image" in str(e):
                logger.warning("CUDA error detected, switching to CPU mode permanently")
                self.device = "cpu"
                if self.model is not None:
                    try:
                        logger.info("Moving model to CPU permanently")
                        self.model = self.model.to("cpu")
                    except Exception as move_err:
                        logger.error(f"Error moving model to CPU: {str(move_err)}")
                        self.model = None
            
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

    def detectItems(self, image_path: str) -> List[str]:
        """
        Detect food items in an image and return a list of item names.
        This method is specifically designed for pantry detection.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of detected item names (strings only)
        """
        try:
            # Use the existing image analysis functionality
            detected_items = self.__call__(image_path)
            
            # Extract just the item names for pantry filtering
            item_names = [item["name"] for item in detected_items if item["name"] != "unknown item"]
            
            logger.info(f"Detected {len(item_names)} pantry items: {item_names}")
            return item_names
            
        except Exception as e:
            logger.error(f"Error detecting pantry items: {e}")
            # Return empty list if detection fails
            return []

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
        
    def clear_list(self):
        """Clear the current shopping list."""
        self.items.clear()
    def get_list(self) -> List[Dict[str, str]]:
        """Get the current shopping list in a structured format."""
        return [{"name": item.name, "quantity": item.quantity} 
                for item in sorted(self.items, key=lambda x: x.name)]
    
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
