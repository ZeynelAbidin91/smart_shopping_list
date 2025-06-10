"""
Test script for the AI Meal Planner feature.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from meal_planner import MealPlannerService, MealPlanRequest

def test_meal_planner():
    """Test the meal planner functionality."""
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
        return False
    
    print("🧪 Testing AI Meal Planner...")
    
    try:
        # Initialize the service
        meal_planner = MealPlannerService()
        print("✅ Meal planner service initialized")
        
        # Create a test request
        request = MealPlanRequest(
            duration=3,
            cuisine_preferences=["Italian", "Mediterranean"],
            dietary_restrictions=["Vegetarian"],
            notes="No mushrooms please"
        )
        
        print("🎯 Generating 3-day vegetarian meal plan...")
        
        # Generate meal plan
        result = meal_planner.generate_meal_plan(request)
        
        if result["success"]:
            print("✅ Meal plan generated successfully!")
            
            # Display the plan
            meal_plan = result["mealPlan"]
            for day_data in meal_plan:
                day = day_data["day"]
                print(f"\n📅 Day {day}:")
                print(f"  🌅 Breakfast: {day_data['breakfast']['name']}")
                print(f"     {day_data['breakfast']['description']}")
                print(f"  ☀️ Lunch: {day_data['lunch']['name']}")
                print(f"     {day_data['lunch']['description']}")
                print(f"  🌙 Dinner: {day_data['dinner']['name']}")
                print(f"     {day_data['dinner']['description']}")
            
            # Test day regeneration
            print("\n🔄 Testing day regeneration for Day 2...")
            
            original_prefs = {
                "cuisine_preferences": ["Italian", "Mediterranean"],
                "dietary_restrictions": ["Vegetarian"],
                "notes": "No mushrooms please"
            }
            
            day_to_regenerate = {
                "day": 2,
                "current_meals": meal_plan[1]  # Day 2 is index 1
            }
            
            regen_result = meal_planner.regenerate_day(original_prefs, day_to_regenerate)
            
            if regen_result["success"]:
                print("✅ Day regeneration successful!")
                print(f"  🌅 New Breakfast: {regen_result['breakfast']['name']}")
                print(f"  ☀️ New Lunch: {regen_result['lunch']['name']}")
                print(f"  🌙 New Dinner: {regen_result['dinner']['name']}")
            else:
                print(f"❌ Day regeneration failed: {regen_result['error']}")
                return False
            
            print("\n🎉 All tests passed! Meal planner is working correctly.")
            return True
            
        else:
            print(f"❌ Meal plan generation failed: {result['error']}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_meal_planner()
    sys.exit(0 if success else 1)
