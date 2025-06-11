"""
AI Meal Planner service using OpenAI GPT-4o mini.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

# Set OpenMP environment variable to handle multiple runtime versions
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class MealPlanRequest:
    """Data class for meal plan request."""
    duration: int
    cuisine_preferences: List[str]
    dietary_restrictions: List[str]
    notes: str = ""

@dataclass
class Meal:
    """Data class for a single meal."""
    name: str
    description: str

@dataclass
class DayMealPlan:
    """Data class for a day's meal plan."""
    day: int
    breakfast: Meal
    lunch: Meal
    dinner: Meal

@dataclass
class ShoppingListItem:
    """Data class for a shopping list item."""
    item: str
    quantity: str
    status: str  # 'needed' or 'in_pantry'

@dataclass
class ShoppingListCategory:
    """Data class for a shopping list category."""
    category: str
    items: List[ShoppingListItem]

class MealPlannerService:
    """Service for generating meal plans using OpenAI GPT-4o mini."""
    
    def __init__(self):
        """Initialize the meal planner service."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set. Please add it to your .env file.")
        
        try:
            logger.info("Initializing OpenAI client...")
            self.client = OpenAI(api_key=api_key)
            self.model = "gpt-4o-mini"
            logger.info("OpenAI client initialized successfully")
            
            # Test the API key with a small request
            logger.info("Testing OpenAI API connection...")
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Test"}
                ],
                max_tokens=5
            )
            logger.info("OpenAI API connection test successful")
            
            # Initialize SmartShoppingList module
            self.smart_shopping_list = SmartShoppingList(self)
            logger.info("SmartShoppingList module initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise ValueError(f"Failed to initialize OpenAI client: {str(e)}")
        
    def generate_meal_plan(self, request: MealPlanRequest) -> Dict[str, Any]:
        """
        Generate a complete meal plan based on user preferences.
        
        Args:
            request: MealPlanRequest containing user preferences
            
        Returns:
            Dict containing the meal plan or error information
        """
        try:
            # Construct the prompt
            prompt = self._build_meal_plan_prompt(request)
            
            # Make API call to OpenAI
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional nutritionist and chef. Generate meal plans that are balanced, delicious, and follow the user's dietary requirements."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            # Parse the response
            content = response.choices[0].message.content.strip()
            meal_plan_data = self._parse_meal_plan_response(content)
            
            return {
                "success": True,
                "mealPlan": meal_plan_data
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            # Retry once with a more explicit prompt
            try:
                return self._retry_meal_plan_generation(request)
            except Exception as retry_error:
                logger.error(f"Retry also failed: {retry_error}")
                return {
                    "success": False,
                    "error": "Failed to generate meal plan. Please try again."
                }
                
        except Exception as e:
            logger.error(f"Error generating meal plan: {e}")
            return {
                "success": False,
                "error": f"Failed to generate meal plan: {str(e)}"
            }
    
    def regenerate_day(self, original_preferences: Dict[str, Any], day_to_regenerate: Dict[str, Any]) -> Dict[str, Any]:
        """
        Regenerate meals for a specific day.
        
        Args:
            original_preferences: The original meal plan preferences
            day_to_regenerate: Information about the day to regenerate
            
        Returns:
            Dict containing the new day's meals or error information
        """
        try:
            # Construct the prompt for day regeneration
            prompt = self._build_day_regeneration_prompt(original_preferences, day_to_regenerate)
            
            # Make API call to OpenAI
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional nutritionist and chef. Generate new meal suggestions that are different from the current ones."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.8,  # Slightly higher temperature for more variety
                max_tokens=800
            )
            
            # Parse the response
            content = response.choices[0].message.content.strip()
            day_meals = self._parse_day_regeneration_response(content)
            
            return {
                "success": True,
                **day_meals
            }
            
        except Exception as e:
            logger.error(f"Error regenerating day: {e}")
            return {
                "success": False,
                "error": f"Failed to regenerate day: {str(e)}"
            }
    
    def generate_shopping_list(self, meal_plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a smart shopping list from a confirmed meal plan.
        
        Args:
            meal_plan: List of daily meal plans
            
        Returns:
            Dict containing the shopping list or error information
        """
        try:
            # Step A: Generate initial shopping list from meal plan
            shopping_list_data = self._generate_initial_shopping_list(meal_plan)
            
            # Step B: Filter with ImageAnalysisTool
            filtered_shopping_list = self._filter_shopping_list_with_pantry(shopping_list_data)
            
            return {
                "success": True,
                "shoppingList": filtered_shopping_list
            }
            
        except Exception as e:
            logger.error(f"Error generating shopping list: {e}")
            return {
                "success": False,
                "error": f"Failed to generate shopping list: {str(e)}"
            }
    
    def _build_meal_plan_prompt(self, request: MealPlanRequest) -> str:
        """Build the prompt for meal plan generation."""
        cuisine_str = ", ".join(request.cuisine_preferences) if request.cuisine_preferences else "Any"
        dietary_str = ", ".join(request.dietary_restrictions) if request.dietary_restrictions else "None"
        
        prompt = f"""Generate a meal plan for {request.duration} days.

Constraints:
- Cuisines: {cuisine_str}
- Dietary needs: {dietary_str}
- Exclusions/Notes: {request.notes}

Output Format:
Return ONLY a valid JSON object. The root should be an array named "mealPlan". Each object in the array represents a day and must contain the keys: day, breakfast, lunch, and dinner. Each meal object should have "name" and "description" keys.

Example: {{"mealPlan": [{{"day": 1, "breakfast": {{"name": "Scrambled Tofu", "description": "Tofu scrambled with turmeric and black salt."}}, "lunch": {{"name": "Lentil Soup", "description": "Hearty red lentil soup with a slice of gluten-free bread."}}, "dinner": {{"name": "Vegetable Risotto", "description": "Creamy risotto with asparagus and peas."}}}}]}}

Make sure the meals are varied, balanced, and follow the dietary constraints. Provide detailed descriptions for each meal."""
        
        return prompt
    
    def _build_day_regeneration_prompt(self, original_preferences: Dict[str, Any], day_to_regenerate: Dict[str, Any]) -> str:
        """Build the prompt for day regeneration."""
        prefs = original_preferences
        current_meals = day_to_regenerate["current_meals"]
        
        cuisine_str = ", ".join(prefs.get("cuisine_preferences", [])) if prefs.get("cuisine_preferences") else "Any"
        dietary_str = ", ".join(prefs.get("dietary_restrictions", [])) if prefs.get("dietary_restrictions") else "None"
        
        prompt = f"""Generate a new breakfast, lunch, and dinner for a single day.

Constraints:
- Cuisines: {cuisine_str}
- Dietary needs: {dietary_str}
- Exclusions/Notes: {prefs.get("notes", "")}

Important: The new meal suggestions MUST be different from the following:
- Breakfast: {current_meals["breakfast"]["name"]}
- Lunch: {current_meals["lunch"]["name"]}
- Dinner: {current_meals["dinner"]["name"]}

Output Format:
Return ONLY a valid JSON object with breakfast, lunch, and dinner keys. Each meal object needs "name" and "description" keys.

Example: {{"breakfast": {{"name": "Avocado Toast", "description": "Avocado on toasted gluten-free bread with chili flakes."}}, "lunch": {{"name": "Caprese Salad", "description": "Fresh tomatoes, mozzarella, and basil."}}, "dinner": {{"name": "Pasta e Fagioli", "description": "Classic Italian pasta and bean soup."}}}}

Make sure the new meals are completely different from the current ones and follow the dietary constraints."""
        
        return prompt
    
    def _parse_meal_plan_response(self, content: str) -> List[Dict[str, Any]]:
        """Parse the meal plan response from OpenAI."""
        try:
            # Try to find JSON in the response
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")
                
            json_str = content[start_idx:end_idx]
            data = json.loads(json_str)
            
            return data.get("mealPlan", [])
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse meal plan response: {e}")
            raise
    
    def _parse_day_regeneration_response(self, content: str) -> Dict[str, Any]:
        """Parse the day regeneration response from OpenAI."""
        try:
            # Try to find JSON in the response
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")
                
            json_str = content[start_idx:end_idx]
            data = json.loads(json_str)
            
            return data
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse day regeneration response: {e}")
            raise
    
    def _retry_meal_plan_generation(self, request: MealPlanRequest) -> Dict[str, Any]:
        """Retry meal plan generation with a more explicit prompt."""
        prompt = f"""You must respond with ONLY valid JSON. No additional text.

Generate {request.duration} days of meals as JSON:

{{"mealPlan": [
  {{"day": 1, "breakfast": {{"name": "meal name", "description": "meal description"}}, "lunch": {{"name": "meal name", "description": "meal description"}}, "dinner": {{"name": "meal name", "description": "meal description"}}}},
  // ... more days
]}}

Requirements:
- Cuisines: {", ".join(request.cuisine_preferences) if request.cuisine_preferences else "Any"}
- Diet: {", ".join(request.dietary_restrictions) if request.dietary_restrictions else "None"}
- Notes: {request.notes}

Return only the JSON object, no other text."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.5,
            max_tokens=2000
        )
        
        content = response.choices[0].message.content.strip()
        meal_plan_data = self._parse_meal_plan_response(content)
        
        return {
            "success": True,
            "mealPlan": meal_plan_data
        }
    
    def _generate_initial_shopping_list(self, meal_plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate initial shopping list from meal plan using LLM.
        
        Args:
            meal_plan: List of daily meal plans
            
        Returns:
            List of shopping list categories
        """
        try:
            # Build the prompt for ingredient extraction
            prompt = self._build_shopping_list_prompt(meal_plan)
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts ingredients from meal plans and creates organized shopping lists."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.3
            )
            
            # Parse the response
            shopping_list = self._parse_shopping_list_response(response.choices[0].message.content)
            
            logger.info(f"Generated initial shopping list with {len(shopping_list)} categories")
            return shopping_list
            
        except Exception as e:
            logger.error(f"Error generating initial shopping list: {e}")
            raise
    
    def _filter_shopping_list_with_pantry(self, shopping_list_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter shopping list by removing items detected in pantry.
        
        Args:
            shopping_list_data: Initial shopping list data
            
        Returns:
            Filtered shopping list with status annotations
        """
        try:
            # Import ImageAnalysisTool here to avoid circular imports
            from llama_agents import ImageAnalysisTool
              # Get pantry items using ImageAnalysisTool
            pantry_items = []
            try:
                image_tool = ImageAnalysisTool()
                
                # In a real implementation, you would pass the path to a pantry image
                # For now, we'll simulate with a placeholder or use a default pantry image
                # pantry_items = image_tool.detectItems("path/to/pantry/image.jpg")
                
                # For testing purposes, we'll use a hybrid approach:
                # Try to detect from a default pantry image if it exists, otherwise use mock
                pantry_image_path = os.path.join(os.path.dirname(__file__), "default_pantry.jpg")
                if os.path.exists(pantry_image_path):
                    pantry_items = image_tool.detectItems(pantry_image_path)
                    logger.info(f"Detected {len(pantry_items)} items from pantry image: {pantry_items}")
                else:
                    # If no pantry image is available, fall back to mock
                    pantry_items = self._mock_detect_pantry_items()
                    logger.info(f"Using mock pantry detection: {pantry_items}")
                    
            except Exception as e:
                logger.warning(f"Failed to detect pantry items: {e}. Using mock detection.")
                pantry_items = self._mock_detect_pantry_items()
            
            # Filter the shopping list
            filtered_list = []
            for category in shopping_list_data:
                filtered_category = {
                    "category": category["category"],
                    "items": []
                }
                
                for item in category["items"]:
                    # Check if item is in pantry (case-insensitive partial matching)
                    item_name = item["item"].lower()
                    is_in_pantry = any(
                        pantry_item.lower() in item_name or item_name in pantry_item.lower()
                        for pantry_item in pantry_items
                    )
                    
                    # Add status field
                    filtered_item = {
                        "item": item["item"],
                        "quantity": item["quantity"],
                        "status": "in_pantry" if is_in_pantry else "needed"
                    }
                    
                    filtered_category["items"].append(filtered_item)
                
                filtered_list.append(filtered_category)
            
            logger.info("Successfully filtered shopping list with pantry items")
            return filtered_list
            
        except Exception as e:
            logger.error(f"Error filtering shopping list: {e}")            # Return original list with all items marked as 'needed'
            for category in shopping_list_data:
                for item in category["items"]:
                    item["status"] = "needed"
            return shopping_list_data
    
    def _mock_detect_pantry_items(self) -> List[str]:
        """
        Fallback function to simulate pantry detection when ImageAnalysisTool fails.
        The main implementation now tries to use ImageAnalysisTool.detectItems() first.
        """
        # This simulates common pantry items that users typically have
        return [
            'salt', 'pepper', 'olive oil', 'garlic', 'onion', 'butter', 
            'flour', 'sugar', 'eggs', 'milk', 'rice', 'pasta'
        ]
    
    def _build_shopping_list_prompt(self, meal_plan: List[Dict[str, Any]]) -> str:
        """
        Build the prompt for shopping list generation.
        
        Args:
            meal_plan: List of daily meal plans
            
        Returns:
            Formatted prompt string
        """
        meal_plan_json = json.dumps(meal_plan, indent=2)
        
        prompt = f"""Analyze the following JSON meal plan and generate a comprehensive, categorized shopping list.

Meal Plan:
{meal_plan_json}

Instructions:
1. Extract every single ingredient required to make all the meals.
2. Consolidate duplicate ingredients and sum their quantities (e.g., if one recipe needs 1 onion and another needs 1 onion, the list should have '2 onions').
3. Group the items into logical categories: Produce, Dairy & Alternatives, Protein, Pantry Staples, Spices & Oils, Other.
4. Provide estimated quantities for each item (e.g., "2 large" onions, "200g" of tofu, "1 cup" of lentils).
5. Be comprehensive but practical - include all necessary ingredients.

Output Format:
Return ONLY a valid JSON object. The root key should be "shoppingList", containing an array of category objects. Each category object must have a "category" name and an array of "items". Each item object must have an "item" name and a "quantity".

Example: {{"shoppingList": [{{"category": "Produce", "items": [{{"item": "Onion", "quantity": "2 large"}}, {{"item": "Garlic", "quantity": "3 cloves"}}]}}, {{"category": "Pantry Staples", "items": [{{"item": "Rice", "quantity": "2 cups"}}]}}]}}

Important: Return ONLY the JSON object, no additional text or explanation."""
        
        return prompt
    
    def _parse_shopping_list_response(self, content: str) -> List[Dict[str, Any]]:
        """
        Parse the shopping list response from OpenAI.
        
        Args:
            content: Response content from OpenAI
            
        Returns:
            List of shopping list categories
        """
        try:
            # Try to find JSON in the response
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")
                
            json_str = content[start_idx:end_idx]
            data = json.loads(json_str)
            
            return data.get("shoppingList", [])
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse shopping list response: {e}")
            raise

class SmartShoppingList:
    """
    Unified module for generating shopping lists from meal plans and filtering
    them based on user's existing inventory through image analysis.
    """
    def __init__(self, meal_planner_service: 'MealPlannerService'):
        """
        Initialize the SmartShoppingList module.
        
        Args:
            meal_planner_service: Instance of MealPlannerService for meal plan processing
        """
        self.meal_planner = meal_planner_service
        self.image_analyzer = None
          # Initialize ImageAnalysisTool
        try:
            from llama_agents import get_image_analyzer
            self.image_analyzer = get_image_analyzer()
            logger.info("✅ ImageAnalysisTool initialized successfully")
        except Exception as e:
            logger.error(f"ImageAnalysisTool initialization failed: {e}")
            self.image_analyzer = None
    
    def generate_list(self, meal_plan_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Step 1: Generate initial comprehensive shopping list from meal plan data.
        
        Args:
            meal_plan_data: List of daily meal plans with breakfast, lunch, dinner
            
        Returns:
            Dict containing the initial shopping list or error information
        """
        try:
            logger.info("Step 1: Generating initial shopping list from meal plan")
            
            # Use the existing meal planner service to generate initial list
            result = self.meal_planner._generate_initial_shopping_list(meal_plan_data)
            
            logger.info(f"Generated initial shopping list with {len(result)} categories")
            return {
                "success": True,
                "shopping_list": result
            }
            
        except Exception as e:
            logger.error(f"Error generating initial shopping list: {e}")
            return {
                "success": False,
                "error": f"Failed to generate shopping list: {str(e)}"
            }
    
    def filter_list(self, initial_shopping_list: List[Dict[str, Any]], user_images: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Step 2: Filter shopping list by removing items detected in user's inventory images.
        
        Args:
            initial_shopping_list: Initial shopping list from generate_list()
            user_images: Single image path or list of image paths for inventory analysis
            
        Returns:
            Dict containing the filtered shopping list
        """
        try:
            logger.info("Step 2: Filtering shopping list based on inventory detection")
            
            if not self.image_analyzer:
                logger.warning("ImageAnalysisTool not available, returning original list")
                return {
                    "success": True,
                    "shopping_list": initial_shopping_list,
                    "removed_items": [],
                    "message": "Image analysis not available, showing full list"
                }
            
            # Handle both single image and multiple images
            if isinstance(user_images, str):
                user_images = [user_images]
              # Detect items from all provided images
            detected_items = []
            for image_path in user_images:
                try:
                    logger.info(f"Analyzing image: {image_path}")
                    items = self.image_analyzer.detectItems(image_path)
                    logger.info(f"Raw detected items from {image_path}: {items}")
                    
                    # Filter out problematic items
                    valid_items = []
                    for item in items:
                        item_clean = item.lower().strip()
                        # Skip items that are too generic or short
                        if (len(item_clean) >= 3 and 
                            item_clean not in ['unknown', 'item', 'food', 'unknown item'] and
                            not item_clean.startswith('unknown')):
                            valid_items.append(item_clean)
                        else:
                            logger.info(f"Filtered out generic/short item: '{item}'")
                    
                    detected_items.extend(valid_items)
                    logger.info(f"Valid items from {image_path}: {valid_items}")
                except Exception as e:
                    logger.warning(f"Failed to analyze image {image_path}: {e}")
            
            # Remove duplicates and normalize detected items
            detected_items = list(set(detected_items))
            logger.info(f"Final detected inventory items: {detected_items} (total: {len(detected_items)})")
            
            # Safety check: if too many items detected, something might be wrong
            if len(detected_items) > 50:
                logger.warning(f"Unusually high number of detected items ({len(detected_items)}). Using fallback approach.")
                detected_items = detected_items[:20]  # Limit to first 20 items
              # Filter the shopping list
            filtered_list = []
            removed_items = []
            
            for category in initial_shopping_list:
                filtered_category = {
                    "category": category["category"],
                    "items": []
                }
                
                for item in category["items"]:
                    item_name = item["item"].lower().strip()
                    
                    # More precise matching logic
                    is_in_inventory = False
                    
                    # Only check if we have detected items (avoid matching when no items detected)
                    if detected_items:
                        # Check for exact matches or meaningful partial matches
                        for detected_item in detected_items:
                            detected_item_clean = detected_item.lower().strip()
                            
                            # Skip empty or very short detected items
                            if len(detected_item_clean) < 3:
                                continue
                                
                            # Exact match
                            if item_name == detected_item_clean:
                                is_in_inventory = True
                                logger.info(f"Exact match: '{item_name}' == '{detected_item_clean}'")
                                break
                            
                            # Meaningful partial match (both directions, but with length check)
                            # Only match if the detected item is substantial (>3 chars) and significantly overlaps
                            if (len(detected_item_clean) >= 4 and 
                                (detected_item_clean in item_name or item_name in detected_item_clean)):
                                # Additional check: ensure the match is meaningful (>50% overlap)
                                overlap_ratio = min(len(detected_item_clean), len(item_name)) / max(len(detected_item_clean), len(item_name))
                                if overlap_ratio > 0.5:
                                    is_in_inventory = True
                                    logger.info(f"Partial match: '{item_name}' <-> '{detected_item_clean}' (overlap: {overlap_ratio:.2f})")
                                    break
                    
                    if is_in_inventory:
                        # Item found in inventory - remove from shopping list
                        removed_items.append({
                            "item": item["item"],
                            "quantity": item["quantity"],
                            "category": category["category"]
                        })
                        logger.info(f"Removed '{item['item']}' - found in inventory")
                    else:
                        # Item not in inventory - keep in shopping list
                        filtered_category["items"].append(item)
                        logger.debug(f"Kept '{item['item']}' - not found in inventory")
                  # Only add category if it has remaining items
                if filtered_category["items"]:
                    filtered_list.append(filtered_category)
            
            logger.info(f"Filtering complete: {len(removed_items)} items removed from {sum(len(cat['items']) for cat in initial_shopping_list)} total items")
            
            # Safety check: if we removed everything, something is wrong
            total_initial_items = sum(len(cat['items']) for cat in initial_shopping_list)
            if len(removed_items) >= total_initial_items:
                logger.warning("All items were removed! This seems incorrect. Returning original list.")
                return {
                    "success": True,
                    "shopping_list": initial_shopping_list,
                    "removed_items": [],
                    "detected_inventory": detected_items,
                    "message": "Safety check: Returned full list to avoid removing all items"
                }
            
            # Another safety check: if we removed more than 80% of items, be cautious
            removal_percentage = len(removed_items) / total_initial_items if total_initial_items > 0 else 0
            if removal_percentage > 0.8:
                logger.warning(f"Removed {removal_percentage:.1%} of items. This seems high. Please review.")
            
            return {
                "success": True,
                "shopping_list": filtered_list,
                "removed_items": removed_items,
                "detected_inventory": detected_items
            }
            
        except Exception as e:
            logger.error(f"Error filtering shopping list: {e}")
            return {
                "success": False,
                "error": f"Failed to filter shopping list: {str(e)}",
                "shopping_list": initial_shopping_list  # Return original on error
            }
    
    def generate_smart_shopping_list(self, meal_plan_data: List[Dict[str, Any]], user_images: Union[str, List[str]] = None) -> Dict[str, Any]:
        """
        Complete workflow: Generate shopping list from meal plan and filter based on inventory.
        
        Args:
            meal_plan_data: List of daily meal plans
            user_images: Optional image(s) for inventory analysis
            
        Returns:
            Dict containing the final filtered shopping list
        """
        try:
            logger.info("Starting complete SmartShoppingList workflow")
            
            # Step 1: Generate initial list
            initial_result = self.generate_list(meal_plan_data)
            if not initial_result["success"]:
                return initial_result
            
            initial_shopping_list = initial_result["shopping_list"]
            
            # Step 2: Filter list if images provided
            if user_images:
                filter_result = self.filter_list(initial_shopping_list, user_images)
                if filter_result["success"]:
                    return {
                        "success": True,
                        "shopping_list": filter_result["shopping_list"],
                        "removed_items": filter_result.get("removed_items", []),
                        "detected_inventory": filter_result.get("detected_inventory", []),
                        "workflow_complete": True
                    }
                else:
                    # Return initial list if filtering failed
                    return {
                        "success": True,
                        "shopping_list": initial_shopping_list,
                        "error": filter_result.get("error"),
                        "workflow_complete": False
                    }
            else:
                # No images provided - return initial list
                logger.info("No inventory images provided, returning initial shopping list")
                return {
                    "success": True,
                    "shopping_list": initial_shopping_list,
                    "removed_items": [],
                    "detected_inventory": [],
                    "workflow_complete": True,
                    "message": "No inventory analysis performed"
                }
                
        except Exception as e:
            logger.error(f"Error in complete SmartShoppingList workflow: {e}")
            return {
                "success": False,
                "error": f"SmartShoppingList workflow failed: {str(e)}"
            }
