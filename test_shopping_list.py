"""
Test script for the shopping list generation feature.
"""

import os
import sys
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from meal_planner import MealPlannerService, MealPlanRequest

def test_shopping_list_generation():
    """Test the shopping list generation functionality."""
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
        return False
    
    print("🧪 Testing Shopping List Generation...")
    
    try:
        # Initialize the service
        meal_planner = MealPlannerService()
        print("✅ Meal planner service initialized")
        
        # Create a test meal plan
        test_meal_plan = [
            {
                "day": 1,
                "breakfast": {
                    "name": "Scrambled Eggs with Toast",
                    "description": "Fluffy scrambled eggs served with buttered whole grain toast"
                },
                "lunch": {
                    "name": "Caesar Salad with Chicken",
                    "description": "Fresh romaine lettuce with grilled chicken, parmesan cheese, and Caesar dressing"
                },
                "dinner": {
                    "name": "Spaghetti Carbonara",
                    "description": "Classic Italian pasta with eggs, bacon, parmesan cheese, and black pepper"
                }
            },
            {
                "day": 2,
                "breakfast": {
                    "name": "Greek Yogurt with Berries",
                    "description": "Creamy Greek yogurt topped with fresh blueberries and honey"
                },
                "lunch": {
                    "name": "Grilled Cheese Sandwich",
                    "description": "Golden grilled cheese sandwich with tomato soup"
                },
                "dinner": {
                    "name": "Chicken Stir Fry",
                    "description": "Tender chicken with mixed vegetables in a savory soy-based sauce, served over rice"
                }
            }
        ]
        
        print("🛒 Generating shopping list from test meal plan...")
        
        # Generate shopping list
        result = meal_planner.generate_shopping_list(test_meal_plan)
        
        if result["success"]:
            print("✅ Shopping list generated successfully!")
            
            # Display the shopping list
            shopping_list = result["shoppingList"]
            print(f"\n📋 Generated Shopping List ({len(shopping_list)} categories):")
            
            total_items = 0
            needed_items = 0
            pantry_items = 0
            
            for category in shopping_list:
                print(f"\n📂 {category['category']}:")
                
                category_needed = [item for item in category['items'] if item['status'] == 'needed']
                category_pantry = [item for item in category['items'] if item['status'] == 'in_pantry']
                
                if category_needed:
                    print("  ✅ Items to Buy:")
                    for item in category_needed:
                        print(f"    - {item['item']} - {item['quantity']}")
                        needed_items += 1
                
                if category_pantry:
                    print("  🏠 Already in Pantry:")
                    for item in category_pantry:
                        print(f"    - {item['item']} - {item['quantity']} (in pantry)")
                        pantry_items += 1
                
                total_items += len(category['items'])
            
            print(f"\n📊 Summary:")
            print(f"  Total items: {total_items}")
            print(f"  Items to buy: {needed_items}")
            print(f"  Items in pantry: {pantry_items}")
            
            # Test the JSON structure
            print(f"\n🔍 Validating JSON structure...")
            
            required_fields = ['category', 'items']
            item_fields = ['item', 'quantity', 'status']
            
            for category in shopping_list:
                for field in required_fields:
                    if field not in category:
                        print(f"❌ Missing field '{field}' in category")
                        return False
                
                for item in category['items']:
                    for field in item_fields:
                        if field not in item:
                            print(f"❌ Missing field '{field}' in item")
                            return False
                    
                    if item['status'] not in ['needed', 'in_pantry']:
                        print(f"❌ Invalid status '{item['status']}' for item {item['item']}")
                        return False
            
            print("✅ JSON structure validation passed")
            
            print("\n🎉 Shopping list generation test passed!")
            return True
            
        else:
            print(f"❌ Shopping list generation failed: {result['error']}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

def test_full_workflow():
    """Test the complete meal plan to shopping list workflow."""
    print("\n🔄 Testing Full Workflow (Meal Plan → Shopping List)...")
    
    try:
        meal_planner = MealPlannerService()
        
        # Step 1: Generate a meal plan
        request = MealPlanRequest(
            duration=2,
            cuisine_preferences=["Italian"],
            dietary_restrictions=["Vegetarian"],
            notes="No mushrooms"
        )
        
        print("1️⃣ Generating meal plan...")
        meal_plan_result = meal_planner.generate_meal_plan(request)
        
        if not meal_plan_result["success"]:
            print(f"❌ Meal plan generation failed: {meal_plan_result['error']}")
            return False
        
        print("✅ Meal plan generated")
        
        # Step 2: Generate shopping list from meal plan
        print("2️⃣ Generating shopping list from meal plan...")
        shopping_list_result = meal_planner.generate_shopping_list(meal_plan_result["mealPlan"])
        
        if not shopping_list_result["success"]:
            print(f"❌ Shopping list generation failed: {shopping_list_result['error']}")
            return False
        
        print("✅ Shopping list generated from meal plan")
        
        # Step 3: Validate the workflow
        meal_plan = meal_plan_result["mealPlan"]
        shopping_list = shopping_list_result["shoppingList"]
        
        print(f"📊 Workflow Summary:")
        print(f"  Meal plan days: {len(meal_plan)}")
        print(f"  Shopping categories: {len(shopping_list)}")
        
        total_shopping_items = sum(len(cat['items']) for cat in shopping_list)
        print(f"  Total shopping items: {total_shopping_items}")
        
        print("\n🎉 Full workflow test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Full workflow test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Running Shopping List Tests...\n")
    
    test1 = test_shopping_list_generation()
    test2 = test_full_workflow()
    
    if test1 and test2:
        print("\n✅ All tests passed! Shopping list feature is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1)
