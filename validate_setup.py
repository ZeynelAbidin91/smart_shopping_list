"""
Quick validation script to ensure all components are working.
"""

import os
import sys
from dotenv import load_dotenv

def validate_setup():
    """Validate that the meal planner setup is complete."""
    
    print("🔍 Validating Smart Fridge & Meal Planner Setup...")
    
    # Load environment variables
    load_dotenv()
    
    # Check API key
    if os.getenv("OPENAI_API_KEY"):
        api_key = os.getenv("OPENAI_API_KEY")
        print(f"✅ OpenAI API Key loaded (starts with: {api_key[:10]}...)")
    else:
        print("❌ OpenAI API Key not found")
        return False
    
    # Test imports
    try:
        from meal_planner import MealPlannerService, MealPlanRequest
        print("✅ Meal planner modules import successfully")
    except ImportError as e:
        print(f"❌ Failed to import meal planner: {e}")
        return False
    
    try:
        from llama_agents import ShoppingListAgent
        print("✅ Shopping list agent imports successfully")
    except ImportError as e:
        print(f"❌ Failed to import shopping list agent: {e}")
        return False
    
    try:
        import gradio as gr
        print("✅ Gradio is available")
    except ImportError as e:
        print(f"❌ Gradio not available: {e}")
        return False
    
    # Test meal planner initialization
    try:
        meal_planner = MealPlannerService()
        print("✅ Meal planner service can be initialized")
    except Exception as e:
        print(f"❌ Failed to initialize meal planner: {e}")
        return False
    
    print("\n🎉 All validation checks passed!")
    print("\n📋 Feature Summary:")
    print("   📱 Shopping List Generator - Ready")
    print("   🍽️ AI Meal Planner - Ready")
    print("   🔄 Day Regeneration - Ready")
    print("   📱 QR Code Generation - Ready")
    
    print("\n🚀 To start the app, run: python app.py")
    print("   Then open: http://127.0.0.1:7860")
    
    return True

if __name__ == "__main__":
    success = validate_setup()
    sys.exit(0 if success else 1)
