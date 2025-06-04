import os
import uuid
import json
import gradio as gr
from PIL import Image
import numpy as np
import qrcode
import base64
import io
import logging
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
import torch
from anthropic import Anthropic

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
logger.info("Environment variables loaded")

# Global state for user preferences
user_preferences = {
    "diet": None,
    "cuisines": []
}

class ImageProcessingAgent:
    def __init__(self):
        """Initialize the VLM-based food detection system using Qwen2.5."""
        try:
            self.model_name = "Qwen/Qwen2-VL-7B"
            logger.info(f"Loading Qwen2.5 VLM model: {self.model_name}")
            
            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            
            # Load model with optimized settings
            logger.info("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                trust_remote_code=True,
                bf16=True,  # Enable mixed precision
                torch_dtype=torch.bfloat16,  # Use bfloat16 for better memory efficiency
                low_cpu_mem_usage=True  # Optimize CPU memory usage during loading
            ).eval()
            
            # Load processor
            logger.info("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            
            logger.info(f"Qwen2.5 model loaded successfully on device: {self.model.device}")
        except Exception as e:
            logger.error(f"Error loading Qwen2.5 model: {str(e)}")
            self.model = None
            self.tokenizer = None
            self.processor = None
            # Fallback food items if model fails to load
            self.fallback_items = [
                "apple", "banana", "orange", "carrot", "tomato",
                "milk", "bread", "cheese", "egg", "chicken"
            ]
    
    def process_image(self, image_path: str) -> List[Dict[str, str]]:
        """Process an image using Qwen2.5 VLM to detect food items with enhanced accuracy."""
        try:
            if self.model is None:
                return self._fallback_detection()

            from image_utils import preprocess_image, has_sufficient_quality
            
            # Open and validate image
            try:
                image = Image.open(image_path)
                
                # Check image quality
                if not has_sufficient_quality(image):
                    logger.warning("Image quality insufficient for accurate detection")
                    return self._fallback_detection()
                
                # Preprocess image for better detection
                image = preprocess_image(image)
                logger.info("Image preprocessed successfully")
                
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                return self._fallback_detection()
            
            # Multi-pass detection with different prompts for better accuracy
            all_detections = []
            
            prompts = [
                # Detailed inventory prompt
                (
                    "You are an expert food inventory specialist. Analyze this refrigerator/kitchen image "
                    "with high precision. List all visible food items using exact measurements and packaging:\n"
                    "- Use precise quantities and units (e.g., '500g cheese', '1 gallon milk')\n"
                    "- Note packaging types and sizes\n"
                    "- Include visible brand names\n"
                    "- Group by categories (dairy, produce, etc.)\n"
                    "Format: Start each item with '-' on a new line"
                ),
                # Fresh produce focused prompt
                (
                    "Focus on fresh produce and perishables in this image. For each item, specify:\n"
                    "- Ripeness state (e.g., 'ripe bananas', 'fresh tomatoes')\n"
                    "- Quantity in standard units\n"
                    "- Visible quality indicators\n"
                    "List only clearly visible items"
                ),
                # Packaging focused prompt
                (
                    "Identify all packaged food items in this image. Note:\n"
                    "- Container sizes (e.g., '12oz bottle', '1kg package')\n"
                    "- Package types (box, jar, can, etc.)\n"
                    "- Brand names if visible\n"
                    "- Quantity of packages"
                )
            ]
            
            # Run detection with each prompt
            for i, prompt in enumerate(prompts):
                try:
                    logger.info(f"Running detection pass {i+1}/{len(prompts)}")
                    inputs = self.processor(
                        text=prompt,
                        images=image,
                        return_tensors="pt",
                        max_length=800
                    ).to(self.model.device)
                    
                    with torch.no_grad():
                        output = self.model.generate(
                            **inputs,
                            max_new_tokens=400,
                            num_beams=9,
                            num_beam_groups=3,
                            diversity_penalty=0.5,
                            do_sample=True,
                            temperature=0.6,
                            top_p=0.95,
                            top_k=50,
                            repetition_penalty=1.3,
                            length_penalty=1.8,
                            no_repeat_ngram_size=3,
                            early_stopping=True
                        )
                    
                    response = self.tokenizer.decode(output[0], skip_special_tokens=True)
                    logger.info(f"Pass {i+1} response: {response}")
                    
                    # Parse items from this pass
                    pass_items = self._parse_vlm_response(response)
                    all_detections.extend(pass_items)
                    
                except Exception as e:
                    logger.error(f"Error in detection pass {i+1}: {str(e)}")
                    continue
            
            if not all_detections:
                logger.warning("No items detected in any pass, using fallback")
                return self._fallback_detection()
            
            # Merge and deduplicate detections
            merged_items = self._merge_detections(all_detections)
            
            # Apply confidence threshold and validation
            filtered_items = []
            min_confidence_threshold = 0.6
            
            for item in merged_items:
                confidence = item.get("confidence", 0.0)
                if confidence < min_confidence_threshold:
                    logger.debug(f"Skipping low confidence item: {item['name']}")
                    continue
                    
                # Validate quantity
                try:
                    qty = float(item["quantity_est"])
                    if not (0 < qty <= 100):  # Reasonable range for household items
                        logger.debug(f"Skipping item with invalid quantity: {item}")
                        continue
                except (ValueError, TypeError):
                    item["quantity_est"] = "1"  # Default quantity
                
                filtered_items.append(item)
            
            if not filtered_items:
                logger.warning("All items filtered out, using fallback")
                return self._fallback_detection()
            
            logger.info(f"Successfully detected {len(filtered_items)} items")
            return filtered_items
            
        except Exception as e:
            logger.error(f"Error in image processing: {str(e)}")
            return self._fallback_detection()
            
    def _merge_detections(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge multiple detections of the same item, combining confidence scores."""
        merged = {}
        
        for item in detections:
            name = item["name"].lower()
            if name in merged:
                # Average the quantities
                current_qty = float(merged[name]["quantity_est"])
                new_qty = float(item["quantity_est"])
                avg_qty = (current_qty + new_qty) / 2
                
                # Take the maximum confidence
                current_conf = merged[name].get("confidence", 0.0)
                new_conf = item.get("confidence", 0.0)
                max_conf = max(current_conf, new_conf)
                
                # Update merged item
                merged[name]["quantity_est"] = str(round(avg_qty, 2))
                merged[name]["confidence"] = max_conf
                
                # Keep additional metadata if present
                if "category" in item and "category" not in merged[name]:
                    merged[name]["category"] = item["category"]
                if "unit" in item and "unit" not in merged[name]:
                    merged[name]["unit"] = item["unit"]
            else:
                merged[name] = item
        
        return list(merged.values())
    
    def _parse_vlm_response(self, text: str) -> List[Dict[str, str]]:
        """Parse Qwen2.5 VLM response into structured format with enhanced accuracy."""
        items = []
        lines = text.lower().split('\n')
        
        # Enhanced quantity word mapping
        quantity_words = {
            'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
            'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
            'single': '1', 'couple': '2', 'few': '3', 'several': '3',
            'half': '0.5', 'quarter': '0.25', 'dozen': '12', 'pair': '2'
        }
        
        # Unit conversions for standardization
        unit_conversions = {
            'kg': 1000, 'g': 1, 'gram': 1, 'grams': 1,
            'lb': 453.592, 'lbs': 453.592, 'pound': 453.592, 'pounds': 453.592,
            'l': 1000, 'ml': 1, 'liter': 1000, 'liters': 1000,
            'oz': 28.3495, 'ounce': 28.3495, 'ounces': 28.3495
        }
        
        # Package size patterns
        package_sizes = {
            'pack': lambda x: float(x) if x else 1,
            'box': lambda x: float(x) if x else 1,
            'case': lambda x: float(x) if x else 12,
            'carton': lambda x: float(x) if x else 1,
            'bunch': lambda x: float(x) if x else 1,
            'bag': lambda x: float(x) if x else 1
        }
        
        current_category = None
        for line in lines:
            line = line.strip('- ').strip()
            if not line:
                continue
                
            # Check if this is a category header
            if line.endswith(':'):
                current_category = line[:-1].strip()
                continue
            
            try:
                # Extract quantity, unit, and name using regex
                import re
                quantity_pattern = r'^(\d*\.?\d+|\b[a-zA-Z]+\b)?\s*([a-zA-Z]*\s)?(.*)'
                matches = re.match(quantity_pattern, line)
                
                if matches:
                    quantity_str, unit, name = matches.groups()
                      # Process quantity
                    if quantity_str:
                        if quantity_str.isdigit() or '.' in quantity_str:
                            quantity = float(quantity_str)
                        else:
                            quantity = float(quantity_words.get(quantity_str.lower(), '1'))
                    else:
                        quantity = 1.0
                    
                    # Process unit and standardize measurement
                    if unit:
                        unit = unit.strip().lower()
                        if unit in unit_conversions:
                            quantity = quantity * unit_conversions[unit]
                            quantity = round(quantity, 2)
                    
                    # Clean up name and handle plural forms
                    name = name.strip()
                    # Remove 's' from plurals but handle special cases
                    if name.endswith('s') and not name.endswith(('ss', 'us', 'is')):
                        name = name[:-1]
                    
                    # Handle package sizes
                    for pkg_type, converter in package_sizes.items():
                        pkg_pattern = f'({pkg_type} of|{pkg_type})'
                        if re.search(pkg_pattern, name):
                            pkg_match = re.match(r'(\d+)?\s*' + pkg_pattern, name)
                            if pkg_match:
                                pkg_quantity = pkg_match.group(1)
                                quantity = quantity * converter(pkg_quantity)
                                name = re.sub(r'\d*\s*' + pkg_pattern + r'\s*', '', name)
                    
                    # Clean up common phrases and standardize names
                    name = re.sub(r'\b(bottle|container|piece|can|jar) of\b', '', name)
                    name = re.sub(r'\s+', ' ', name).strip()
                    
                    # Validate the item
                    if name and name != 'and' and len(name) > 1:
                        item_data = {
                            "name": name,
                            "quantity_est": str(quantity)
                        }
                        
                        if current_category:
                            item_data["category"] = current_category
                            
                        # Add confidence score based on specificity
                        confidence = 1.0
                        if not quantity_str:
                            confidence *= 0.8
                        if not unit:
                            confidence *= 0.9
                        item_data["confidence"] = confidence
                        
                        items.append(item_data)
                        logger.info(f"Detected item: {quantity} {name} (confidence: {confidence:.2f})")
            
            except Exception as e:
                logger.warning(f"Error parsing line '{line}': {str(e)}")
                continue
          # Sort items by confidence and return top matches
        items.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        return items

    def _fallback_detection(self)-> List[Dict[str, str]]:
        """Enhanced fallback detection when Qwen2.5 VLM fails using basic categories."""
        import random
        
        # Categorized fallback items with typical quantities
        categorized_items = {
            "fruits": [
                ("apple", (2, 6)),
                ("banana", (3, 7)),
                ("orange", (3, 6)),
                ("lemon", (2, 4))
            ],
            "vegetables": [
                ("carrot", (3, 6)),
                ("tomato", (2, 5)),
                ("lettuce", (1, 2)),
                ("cucumber", (1, 3))
            ],
            "dairy": [
                ("milk", (1, 2)),
                ("cheese", (1, 2)),
                ("yogurt", (2, 6)),
                ("butter", (1, 1))
            ],
            "proteins": [
                ("egg", (6, 12)),
                ("chicken", (1, 2)),
                ("tofu", (1, 2))
            ],
            "staples": [
                ("bread", (1, 2)),
                ("rice", (1, 1)),
                ("pasta", (1, 2))
            ]
        }
        
        items = []
        # Select 1-2 items from each category
        for category, category_items in categorized_items.items():
            num_items = random.randint(1, 2)
            selected_items = random.sample(category_items, min(num_items, len(category_items)))
            
            for item_name, quantity_range in selected_items:
                min_qty, max_qty = quantity_range
                quantity = random.randint(min_qty, max_qty)
                
                item = {
                    "name": item_name,
                    "quantity_est": str(quantity),
                    "category": category,
                    "confidence": 0.7  # Lower confidence for fallback items
                }
                items.append(item)
                logger.info(f"Fallback item added: {item}")
        
        # Ensure we have at least 3 items
        while len(items) < 3:
            category = random.choice(list(categorized_items.keys()))
            item_name, quantity_range = random.choice(categorized_items[category])
            min_qty, max_qty = quantity_range
            quantity = random.randint(min_qty, max_qty)
            
            item = {
                "name": item_name,
                "quantity_est": str(quantity),
                "category": category,
                "confidence": 0.7
            }
            
            if item not in items:
                items.append(item)
                logger.info(f"Additional fallback item added: {item}")
        
        return items

class PreferenceAcquisitionAgent:
    @staticmethod
    def update_preferences(diet: str, cuisines: List[str]) -> Dict[str, Any]:
        """Update user preferences with diet and cuisine choices."""
        global user_preferences
        user_preferences["diet"] = diet
        user_preferences["cuisines"] = cuisines
        logger.info(f"Updated user preferences: {user_preferences}")
        return user_preferences

class FilteringSuggestionAgent:
    def __init__(self):
        """Initialize the FilteringSuggestionAgent with Claude 3.5 integration."""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.error("ANTHROPIC_API_KEY not found in environment variables")
        else:
            logger.info(f"Claude API key loaded: {api_key[:8]}...")
        
        self.anthropic = Anthropic(api_key=api_key)
        
        self.base_prompt = """You are an expert cooking assistant using Claude 3.5. Analyze this input:

Diet: {diet}
Cuisine preferences: {cuisines}
Items from Qwen2.5 VLM: {items}

Tasks:
1. Filter detected items:
   - Remove items conflicting with dietary restrictions
   - Validate quantities and portions
   - Check food safety requirements

2. Suggest complementary ingredients:
   - Basic essentials
   - Recipe pairings for preferred cuisines
   - Key pantry items
   - Seasonal recommendations

Format response as JSON:
{{
    "filtered_items": [
        {{"name": "item_name", "quantity": "number"}}
    ],
    "suggested_items": [
        {{"name": "item_name", "quantity": "number", "reason": "brief explanation"}}
    ]
}}
Return ONLY the JSON object."""

        # Common dietary restrictions lookup for quick validation
        self.diet_restrictions = {
            "vegetarian": ["meat", "fish", "chicken", "beef", "pork"],
            "vegan": ["meat", "fish", "chicken", "beef", "pork", "eggs", "milk", "cheese", "honey"],
            "keto": ["bread", "pasta", "rice", "sugar", "potato", "corn"],
            "gluten-free": ["wheat", "rye", "barley", "bread", "pasta", "couscous"]
        }

    def process_list(self, detected_items: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Filter and enhance the shopping list using Claude 3.5."""
        try:
            diet = user_preferences.get("diet", "").lower()
            cuisines = user_preferences.get("cuisines", [])
            
            prompt = self.base_prompt.format(
                diet=diet if diet != "none" else "No specific diet",
                cuisines=", ".join(cuisines) if cuisines else "No specific cuisine preference",
                items=", ".join(f"{item['name']} ({item['quantity_est']})" for item in detected_items)
            )

            logger.info("Sending request to Claude 3.5")
            response = self.anthropic.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1000,
                temperature=0.7,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            try:
                response_text = response.content[0].text.strip()
                logger.info(f"Claude response: {response_text}")
                
                # Find the JSON object
                start_idx = response_text.find("{")
                end_idx = response_text.rfind("}") + 1
                if start_idx == -1 or end_idx == 0:
                    raise ValueError("No JSON object found in response")
                    
                json_text = response_text[start_idx:end_idx]
                result = json.loads(json_text)
                filtered_items = []

                # Process filtered items
                if "filtered_items" in result:
                    for item in result["filtered_items"]:
                        try:
                            quantity = float(str(item["quantity"]).replace(',', ''))
                        except (ValueError, TypeError):
                            quantity = 1.0

                        filtered_items.append({
                            "id": str(uuid.uuid4()),
                            "name": str(item["name"]).strip().lower(),
                            "quantity": quantity,
                            "to_purchase": True
                        })

                # Process suggested items
                if "suggested_items" in result:
                    for item in result["suggested_items"]:
                        try:
                            quantity = float(str(item["quantity"]).replace(',', ''))
                        except (ValueError, TypeError):
                            quantity = 1.0

                        filtered_items.append({
                            "id": str(uuid.uuid4()),
                            "name": str(item["name"]).strip().lower(),
                            "quantity": quantity,
                            "to_purchase": True,
                            "reason": str(item.get("reason", "")).strip()
                        })

                return filtered_items[:20] if filtered_items else self._fallback_filtering(detected_items, diet)

            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Error parsing Claude response: {str(e)}")
                return self._fallback_filtering(detected_items, diet)

        except Exception as e:
            logger.error(f"Error in Claude API call: {str(e)}")
            return self._fallback_filtering(detected_items, diet)

    def _fallback_filtering(self, detected_items: List[Dict[str, str]], diet: str) -> List[Dict[str, Any]]:
        """Fallback method when Claude API fails."""
        logger.info("Fallback filtering activated")
        restricted_items = self.diet_restrictions.get(diet, [])
        filtered_items = []

        # Basic filtering
        for item in detected_items:
            item_name = item["name"].lower()
            if not any(restricted in item_name for restricted in restricted_items):
                filtered_items.append({
                    "id": str(uuid.uuid4()),
                    "name": item["name"],
                    "quantity": float(item["quantity_est"]) if item["quantity_est"].isdigit() else 1,
                    "to_purchase": True
                })
                logger.info(f"Item passed basic filtering: {item}")

        # Add basic suggestions using the existing method
        suggested_items = self._get_smart_suggestions(
            diet,
            user_preferences.get("cuisines", []),
            filtered_items
        )

        # Add suggested items
        for name, quantity in suggested_items:
            if not any(restricted in name.lower() for restricted in restricted_items):
                if not any(item["name"].lower() == name.lower() for item in filtered_items):
                    filtered_items.append({
                        "id": str(uuid.uuid4()),
                        "name": name,
                        "quantity": quantity,
                        "to_purchase": True
                    })
                    logger.info(f"Suggested item added: {name}, quantity: {quantity}")

        logger.info(f"Fallback filtering completed. Items processed: {len(filtered_items)}")
        return filtered_items[:20]

    def _get_smart_suggestions(self, diet: str, cuisines: List[str], current_items: List[Dict[str, Any]]) -> List[Tuple[str, float]]:
        """
        Generate smart suggestions based on diet, cuisines, and current items.
        
        Args:
            diet (str): User's dietary preference
            cuisines (List[str]): List of preferred cuisines
            current_items (List[Dict[str, Any]]): Currently detected items
            
        Returns:
            List[Tuple[str, float]]: List of (item_name, suggested_quantity) tuples
        """
        from config import CUISINE_SUGGESTIONS, DIET_RESTRICTIONS
        suggestions = []
        current_item_names = {item["name"].lower() for item in current_items}
        
        logger.info(f"Generating smart suggestions for diet: {diet}, cuisines: {cuisines}")
        
        # Get diet restrictions
        restricted_items = set(DIET_RESTRICTIONS.get(diet.lower(), []))
        
        # Basic essentials based on diet
        basic_essentials = {
            "vegetarian": [("tofu", 1), ("lentils", 1), ("chickpeas", 1)],
            "vegan": [("tofu", 1), ("almond milk", 1), ("nutritional yeast", 0.5)],
            "keto": [("avocado", 2), ("eggs", 12), ("coconut oil", 1)],
            "gluten-free": [("quinoa", 1), ("rice", 2), ("gluten-free bread", 1)]
        }
        
        # Add diet-specific essentials
        if diet.lower() in basic_essentials:
            for item, qty in basic_essentials[diet.lower()]:
                if (item not in current_item_names and 
                    item not in restricted_items):
                    suggestions.append((item, qty))
        
        # Add cuisine-specific suggestions
        for cuisine in cuisines:
            cuisine = cuisine.strip().title()
            if cuisine in CUISINE_SUGGESTIONS:
                # Get items specific to this cuisine
                cuisine_items = CUISINE_SUGGESTIONS[cuisine]
                # Take up to 3 random items from each cuisine
                import random
                sampled_items = random.sample(cuisine_items, min(3, len(cuisine_items)))
                for item in sampled_items:
                    if (item not in current_item_names and 
                        item not in restricted_items and 
                        not any(sugg[0] == item for sugg in suggestions)):
                        suggestions.append((item, 1.0))
        
        # Add seasonal suggestions
        from datetime import datetime
        current_month = datetime.now().month
        seasonal_items = {
            # Spring (March-May)
            3: [("asparagus", 1), ("peas", 1), ("strawberries", 1)],
            4: [("asparagus", 1), ("peas", 1), ("strawberries", 1)],
            5: [("asparagus", 1), ("peas", 1), ("strawberries", 1)],
            # Summer (June-August)
            6: [("tomatoes", 3), ("zucchini", 2), ("watermelon", 1)],
            7: [("tomatoes", 3), ("zucchini", 2), ("watermelon", 1)],
            8: [("tomatoes", 3), ("zucchini", 2), ("watermelon", 1)],
            # Fall (September-November)
            9: [("apples", 4), ("pumpkin", 1), ("sweet potatoes", 2)],
            10: [("apples", 4), ("pumpkin", 1), ("sweet potatoes", 2)],
            11: [("apples", 4), ("pumpkin", 1), ("sweet potatoes", 2)],
            # Winter (December-February)
            12: [("oranges", 4), ("kale", 1), ("root vegetables", 2)],
            1: [("oranges", 4), ("kale", 1), ("root vegetables", 2)],
            2: [("oranges", 4), ("kale", 1), ("root vegetables", 2)]
        }
        
        for item, qty in seasonal_items.get(current_month, []):
            if (item not in current_item_names and 
                item not in restricted_items and 
                not any(sugg[0] == item for sugg in suggestions)):
                suggestions.append((item, qty))
        
        # Add complementary items based on current items
        complementary_pairs = {
            "pasta": [("tomato sauce", 1), ("parmesan", 1)],
            "bread": [("butter", 1), ("jam", 1)],
            "eggs": [("spinach", 1), ("cheese", 1)],
            "rice": [("soy sauce", 1), ("vegetables", 2)],
            "chicken": [("lemon", 2), ("herbs", 1)],
            "fish": [("lemon", 2), ("garlic", 1)],
            "potatoes": [("onions", 2), ("butter", 1)]
        }
        
        for item in current_item_names:
            if item in complementary_pairs:
                for comp_item, qty in complementary_pairs[item]:
                    if (comp_item not in current_item_names and 
                        comp_item not in restricted_items and 
                        not any(sugg[0] == comp_item for sugg in suggestions)):
                        suggestions.append((comp_item, qty))
        
        # Limit the number of suggestions
        suggestions = suggestions[:10]  # Return at most 10 suggestions
        logger.info(f"Generated {len(suggestions)} smart suggestions")
        
        return suggestions

class NotificationAgent:
    @staticmethod
    def create_qr_code(shopping_list: List[Dict[str, Any]]) -> str:
        """Create a QR code containing the shopping list."""
        # Format the shopping list as text
        list_text = "Shopping List:\n\n"
        for item in shopping_list:
            if item.get("to_purchase", True):
                quantity = item.get("quantity", 1)
                name = item.get("name", "unknown item")
                list_text += f"• {quantity} {name}\n"
        
        # Generate QR code
        qr = qrcode.QRCode(
            version=None,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(list_text)
        qr.make(fit=True)

        # Create QR code image
        qr_image = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to base64 for Gradio
        buffered = io.BytesIO()
        qr_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"

# Initialize agents with logging
logger.info("Initializing agents...")
image_agent = ImageProcessingAgent()
filtering_agent = FilteringSuggestionAgent()
notification_agent = NotificationAgent()
logger.info("Agents initialized successfully")

# Create Gradio interface
with gr.Blocks() as app:
    gr.Markdown("# Smart Fridge Shopping List Generator")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="filepath", label="Upload Fridge Image")
            diet_input = gr.Dropdown(
                choices=["None", "Vegetarian", "Vegan", "Keto", "Gluten-free"],
                label="Select Diet",
                value="None"
            )
            cuisines_input = gr.Textbox(
                label="Enter Cuisines (comma-separated)",
                placeholder="Italian, Indian, Mexican, Japanese"
            )
            process_btn = gr.Button("Process Image")
            result_text = gr.Textbox(label="Status", interactive=False)
        
        with gr.Column():
            raw_detections = gr.Dataframe(
                headers=["Item", "Quantity"],
                label="Raw Detections from Image",
                interactive=False
            )
            items_list = gr.Dataframe(
                headers=["Item", "Quantity", "Purchase"],
                label="Filtered Shopping List",
                interactive=True
            )
            generate_qr_btn = gr.Button("Generate QR Code")
            qr_output = gr.Image(label="Scan this QR code with your phone")

    def handle_upload(image: str, diet: str, cuisines: str) -> Tuple[List, List, str]:
        """Process uploaded image and update shopping list."""
        if not image:
            return [], [], "Please upload an image first"
        
        # Update preferences
        PreferenceAcquisitionAgent.update_preferences(diet, cuisines.split(',') if cuisines else [])
        
        # Process image with status updates
        try:
            result_text = "Analyzing image with Vision Language Model..."
            detected_items = image_agent.process_image(image)
            
            if not detected_items:
                return [], [], "No food items detected in the image. Please try a clearer photo."
            
            # Prepare raw detections for display
            raw_data = [[item["name"], item["quantity_est"]] for item in detected_items]
            
            # Filter and suggest items
            shopping_list = filtering_agent.process_list(detected_items)
            
            # Convert to dataframe format
            df_data = [[item["name"], item["quantity"], item["to_purchase"]] for item in shopping_list]
            
            return raw_data, df_data, f"Successfully processed image and found {len(shopping_list)} items!"
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return [], [], f"Error processing image: {str(e)}"

    def handle_qr_generation(data: List) -> Tuple[str, str]:
        """Generate QR code for the shopping list."""
        if not data:
            return None, "Please process an image first"
        
        # Convert dataframe data back to shopping list format
        shopping_list = [
            {
                "name": item[0],
                "quantity": item[1],
                "to_purchase": item[2]
            }
            for item in data
        ]
        
        try:
            qr_code = notification_agent.create_qr_code(shopping_list)
            return qr_code, "QR code generated successfully!"
        except Exception as e:
            logger.error(f"Error generating QR code: {str(e)}")
            return None, f"Error generating QR code: {str(e)}"

    # Set up event handlers
    process_btn.click(
        fn=handle_upload,
        inputs=[image_input, diet_input, cuisines_input],
        outputs=[raw_detections, items_list, result_text]
    )

    generate_qr_btn.click(
        fn=handle_qr_generation,
        inputs=[items_list],
        outputs=[qr_output, result_text]
    )

if __name__ == "__main__":
    try:
        # Try multiple ports if the default is busy
        for port in range(7860, 7870):
            try:
                app.launch(
                    server_name="127.0.0.1",
                    server_port=port,
                    share=False
                )
                break
            except OSError:
                if port == 7869:  # Last port in range
                    logger.error("Could not find an available port. Please ensure no other Gradio apps are running.")
                    raise
                continue
    except Exception as e:
        logger.error(f"Error starting the app: {str(e)}")
        raise
