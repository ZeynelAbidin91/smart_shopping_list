"""
AI Meal Planner service using OpenAI GPT-4o mini.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
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

class MealPlannerService:
    """Service for generating meal plans using OpenAI GPT-4o mini."""
    
    def __init__(self):
        """Initialize the meal planner service."""
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o-mini"
        
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
