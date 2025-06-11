import os

# Set OpenMP environment variable to handle multiple runtime versions - MUST be before any other imports
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Hugging Face Spaces environment detection
HF_SPACES = os.getenv("SPACE_ID") is not None
if HF_SPACES:
    print("🤗 Running on Hugging Face Spaces")
    # Set HF-specific optimizations
    os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers_cache'
    os.environ['HF_HOME'] = '/tmp/hf_cache'
    
import torch

# Check CUDA availability and set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}")
    if HF_SPACES:
        print("🚀 GPU acceleration enabled on HF Spaces")

import json
import gradio as gr
import qrcode
import io
import base64
import logging
from typing import List, Dict, Any, Tuple, Union
from dotenv import load_dotenv

# Import our LlamaIndex-based agents
from llama_agents import ShoppingListAgent
from meal_planner import MealPlannerService, MealPlanRequest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
logger.info("Environment variables loaded")

# Check OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.warning("OPENAI_API_KEY environment variable is not set. Meal planning feature will not be available.")
    print("\n⚠️ WARNING: OpenAI API key not found in .env file!")
    print("⚠️ Please add your API key to the .env file to enable meal planning.")
    print("⚠️ Continuing with limited functionality (image analysis only)...")
    HAS_OPENAI_API_KEY = False
else:
    logger.info(f"OpenAI API key found (length: {len(openai_api_key)})")
    HAS_OPENAI_API_KEY = True

# Global state for user preferences
user_preferences = {
    "diet": None,
    "cuisines": []
}

# Initialize the shopping list agent and meal planner
shopping_list_agent = ShoppingListAgent()

try:
    meal_planner_service = MealPlannerService()
    MEAL_PLANNING_ENABLED = True
    logger.info("Meal planner service initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize meal planner service: {str(e)}")
    print(f"\n⚠️ ERROR: Could not initialize meal planner: {str(e)}")
    print("⚠️ Meal planning feature will be disabled")
    print("⚠️ Please check your OpenAI API key in the .env file")
    MEAL_PLANNING_ENABLED = False
    # Create a dummy meal planner service that returns empty results
    class DummyMealPlannerService:
        def generate_meal_plan(self, *args, **kwargs):
            return {"error": "Meal planning is not available. Please check your OpenAI API key."}
    meal_planner_service = DummyMealPlannerService()

# Global state for meal plans
meal_plan_state = {
    "current_plan": None,
    "original_preferences": None,
    "current_shopping_list": None,
    "plan_confirmed": False,
    "plan_generation_status": "not_started"  # not_started, generating, ready, confirmed
}

def process_image_and_generate_list(image, diet: str, cuisines: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Process an image and generate a shopping list using LlamaIndex agent."""
    try:
        # Save the temporary image
        temp_image_path = "temp_image.jpg"
        image.save(temp_image_path)
        
        # Parse cuisine preferences
        cuisine_list = [c.strip() for c in cuisines.split(",") if c.strip()] if cuisines else []
        
        # Update preferences
        preferences = {
            "diet": diet.lower() if diet else "none",
            "cuisines": cuisine_list
        }
        
        # Process with LlamaIndex agent
        result = shopping_list_agent.process_image(temp_image_path, preferences)
        detected_items = result['detected_items']
        shopping_list = result['final_list']
        
        # Clean up
        os.remove(temp_image_path)
        
        # Format both lists for display
        formatted_list = "Detected Items:\n"
        for item in detected_items:
            formatted_list += f"• {item['name']}"
            if 'quantity_est' in item:
                formatted_list += f" ({item['quantity_est']})"
            formatted_list += "\n"
        
        formatted_list += "\nFinal Shopping List:\n"
        formatted_list += format_shopping_list(shopping_list)
        
        # Generate QR code
        qr_code = generate_qr_code(shopping_list)
        
        return qr_code, formatted_list
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return None, []

def format_shopping_list(items: List[Dict[str, Any]]) -> str:
    """Format the shopping list items for display."""
    if not items:
        return "No items detected. Please try again with a clearer image."
    
    formatted = []
    for item in items:
        line = f"• {item['name'].title()}: {item['quantity']}"
        if "reason" in item:
            line += f" ({item['reason']})"
        formatted.append(line)
    
    return "\n".join(formatted)

def generate_qr_code(items: List[Dict[str, Any]]) -> str:
    """Generate a QR code for the shopping list that displays nicely on smartphones."""
    try:
        if not items:
            return None
            
        # Create a nicely formatted shopping list for smartphones
        formatted_text = "🛒 SMART SHOPPING LIST\n"
        formatted_text += "=" * 25 + "\n\n"
        
        # Group items by category
        categories = {}
        for item in items:
            category = item.get("category", "Other")
            if category not in categories:
                categories[category] = []
            categories[category].append(item)
        
        # Format each category
        for category, category_items in categories.items():
            formatted_text += f"📂 {category.upper()}\n"
            formatted_text += "-" * len(category) + "\n"
            for item in category_items:
                formatted_text += f"☐ {item['name']} ({item['quantity']})\n"
            formatted_text += "\n"
        
        # Add footer
        from datetime import datetime
        formatted_text += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        formatted_text += "Total items: " + str(len(items)) + "\n"
        formatted_text += "\n💡 Tap checkboxes as you shop!"
        
        # Create QR code with larger size for better smartphone scanning
        qr = qrcode.QRCode(
            version=None,  # Auto-size
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=10,
            border=4
        )
        qr.add_data(formatted_text)
        qr.make(fit=True)
        
        # Create the image
        qr_image = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to base64
        buffered = io.BytesIO()
        qr_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        logger.error(f"Error generating QR code: {str(e)}")
        return None

# Meal Planner Functions
def generate_meal_plan(duration: int, cuisine_prefs: List[str], dietary_restrictions: List[str], notes: str) -> Tuple[str, str, Union[dict, bool]]:
    """Generate a meal plan based on user preferences."""
    try:
        # Reset state
        meal_plan_state["plan_confirmed"] = False
        meal_plan_state["plan_generation_status"] = "generating"
        meal_plan_state["current_shopping_list"] = None
        
        # Validate duration
        if not duration or duration < 1:
            meal_plan_state["plan_generation_status"] = "not_started"
            return "❌ Error: Duration must be at least 1 day", "", gr.update(visible=False)
        
        # Parse preferences - they come as lists from CheckboxGroup
        cuisine_list = cuisine_prefs if isinstance(cuisine_prefs, list) else []
        dietary_list = dietary_restrictions if isinstance(dietary_restrictions, list) else []
        
        # Create request
        request = MealPlanRequest(
            duration=duration,
            cuisine_preferences=cuisine_list,
            dietary_restrictions=dietary_list,
            notes=notes or ""
        )
        
        # Store original preferences for regeneration
        meal_plan_state["original_preferences"] = {
            "cuisine_preferences": cuisine_list,
            "dietary_restrictions": dietary_list,
            "notes": notes or ""
        }
        
        # Generate meal plan
        result = meal_planner_service.generate_meal_plan(request)
        
        if not result["success"]:
            meal_plan_state["plan_generation_status"] = "not_started"
            return f"❌ Error: {result['error']}", "", gr.update(visible=False)
        
        # Store the meal plan
        meal_plan_state["current_plan"] = result["mealPlan"]
        meal_plan_state["plan_generation_status"] = "ready"
        
        # Format for display
        formatted_plan = format_meal_plan_display(result["mealPlan"])
        
        return "✅ Meal plan generated successfully! Review and confirm to create shopping list.", formatted_plan, gr.update(visible=True)
        
    except Exception as e:
        logger.error(f"Error generating meal plan: {e}")
        meal_plan_state["plan_generation_status"] = "not_started"
        return f"❌ Error: {str(e)}", "", gr.update(visible=False)

def confirm_meal_plan() -> Tuple[str, Union[dict, bool]]:
    """Confirm the current meal plan and enable shopping list generation."""
    try:
        if meal_plan_state["plan_generation_status"] != "ready":
            return "❌ Error: No meal plan available to confirm. Please generate a meal plan first.", gr.update(visible=False)
        
        if not meal_plan_state["current_plan"]:
            return "❌ Error: No meal plan data found.", gr.update(visible=False)
        
        # Mark plan as confirmed
        meal_plan_state["plan_confirmed"] = True
        meal_plan_state["plan_generation_status"] = "confirmed"
        
        return "✅ Meal plan confirmed! You can now generate your shopping list.", gr.update(visible=True)
        
    except Exception as e:
        logger.error(f"Error confirming meal plan: {e}")
        return f"❌ Error: {str(e)}", gr.update(visible=False)

def generate_shopping_list_from_meal_plan() -> Tuple[str, str]:
    """Generate a shopping list from the confirmed meal plan."""
    try:
        if not meal_plan_state["plan_confirmed"]:
            return "❌ Error: Please confirm your meal plan first before generating a shopping list.", ""
        
        if not meal_plan_state["current_plan"]:
            return "❌ Error: No meal plan available. Please generate and confirm a meal plan first.", ""
        
        # Generate shopping list
        result = meal_planner_service.generate_shopping_list(meal_plan_state["current_plan"])
        
        if not result["success"]:
            return f"❌ Error: {result['error']}", ""
        
        # Store the shopping list
        meal_plan_state["current_shopping_list"] = result["shoppingList"]
        
        # Format for display
        formatted_shopping_list = format_shopping_list_display(result["shoppingList"])
        
        return "✅ Shopping list generated successfully! Check the Smart Shopping List tab.", formatted_shopping_list
        
    except Exception as e:
        logger.error(f"Error generating shopping list: {e}")
        return f"❌ Error: {str(e)}", ""

def generate_smart_shopping_list_with_inventory(inventory_images=None) -> Tuple[str, str]:
    """Generate a smart shopping list using the new SmartShoppingList module with inventory filtering."""
    try:
        if not meal_plan_state["plan_confirmed"]:
            return "❌ Error: Please confirm your meal plan first before generating a shopping list.", ""
        
        if not meal_plan_state["current_plan"]:
            return "❌ Error: No meal plan available. Please generate and confirm a meal plan first.", ""
        
        # Use the new SmartShoppingList module
        smart_list = meal_planner_service.smart_shopping_list
        
        # Prepare image paths if provided
        image_paths = []
        if inventory_images:
            if isinstance(inventory_images, list):
                for i, img in enumerate(inventory_images):
                    if img is not None:
                        temp_path = f"temp_inventory_{i}.jpg"
                        img.save(temp_path)
                        image_paths.append(temp_path)
            else:
                temp_path = "temp_inventory.jpg"
                inventory_images.save(temp_path)
                image_paths = [temp_path]
        
        # Generate smart shopping list
        result = smart_list.generate_smart_shopping_list(
            meal_plan_data=meal_plan_state["current_plan"],
            user_images=image_paths if image_paths else None
        )
        
        # Clean up temporary files
        for path in image_paths:
            try:
                os.remove(path)
            except:
                pass
        
        if not result["success"]:
            return f"❌ Error: {result['error']}", ""
        
        # Store the shopping list and removed items info
        meal_plan_state["current_shopping_list"] = result["shopping_list"]
        meal_plan_state["last_removed_items"] = result.get("removed_items", [])
        meal_plan_state["detected_inventory"] = result.get("detected_inventory", [])
        
        # Format for display
        formatted_shopping_list = format_smart_shopping_list_display(result)
        
        # Create status message with filtering info
        status_msg = "✅ Smart shopping list generated successfully!"
        if result.get("removed_items"):
            removed_count = len(result["removed_items"])
            status_msg += f" {removed_count} items removed based on your inventory."
        elif image_paths:
            status_msg += " No items were filtered from your inventory."
        
        return status_msg, formatted_shopping_list
        
    except Exception as e:
        logger.error(f"Error generating smart shopping list: {e}")
        return f"❌ Error: {str(e)}", ""

def format_meal_plan_display(meal_plan: List[Dict[str, Any]]) -> str:
    """Format the meal plan for display."""
    if not meal_plan:
        return "No meal plan available."
    
    formatted = []
    for day_data in meal_plan:
        day_num = day_data["day"]
        formatted.append(f"## 📅 Day {day_num}")
        formatted.append("")
        
        # Breakfast
        breakfast = day_data["breakfast"]
        formatted.append(f"### 🌅 Breakfast: {breakfast['name']}")
        formatted.append(f"_{breakfast['description']}_")
        formatted.append("")
        
        # Lunch
        lunch = day_data["lunch"]
        formatted.append(f"### ☀️ Lunch: {lunch['name']}")
        formatted.append(f"_{lunch['description']}_")
        formatted.append("")
        
        # Dinner
        dinner = day_data["dinner"]
        formatted.append(f"### 🌙 Dinner: {dinner['name']}")
        formatted.append(f"_{dinner['description']}_")
        formatted.append("")
        formatted.append("---")
        formatted.append("")
    
    return "\n".join(formatted)

def format_shopping_list_display(shopping_list: List[Dict[str, Any]]) -> str:
    """Format the shopping list for display."""
    if not shopping_list:
        return "No shopping list available."
    
    formatted = []
    formatted.append("# 🛒 Smart Shopping List")
    formatted.append("")
    
    needed_items_count = 0
    pantry_items_count = 0
    
    for category in shopping_list:
        category_name = category["category"]
        items = category["items"]
        
        if not items:
            continue
            
        # Count items by status
        needed_items_in_category = [item for item in items if item.get("status", "needed") == "needed"]
        pantry_items_in_category = [item for item in items if item.get("status", "needed") == "in_pantry"]
        
        needed_items_count += len(needed_items_in_category)
        pantry_items_count += len(pantry_items_in_category)
        
        # Add category header
        formatted.append(f"## 📂 {category_name}")
        formatted.append("")
        
        # Add needed items first
        if needed_items_in_category:
            formatted.append("### ✅ Items to Buy:")
            for item in needed_items_in_category:
                formatted.append(f"- [ ] **{item['item']}** - {item['quantity']}")
            formatted.append("")
        
        # Add pantry items
        if pantry_items_in_category:
            formatted.append("### 🏠 Already in Pantry:")
            for item in pantry_items_in_category:
                formatted.append(f"- [x] ~~{item['item']} - {item['quantity']}~~ **(in pantry)**")
            formatted.append("")
        
        formatted.append("---")
        formatted.append("")
    
    # Add summary
    if pantry_items_count > 0:
        summary = f"**Summary:** {needed_items_count} items to buy, {pantry_items_count} items already in pantry"
    else:
        summary = f"**Summary:** {needed_items_count} total items"
    formatted.insert(2, summary)
    formatted.insert(3, "")
    
    return "\n".join(formatted)

def format_smart_shopping_list_display(result: Dict[str, Any]) -> str:
    """Format the smart shopping list with inventory filtering information."""
    shopping_list = result.get("shopping_list", [])
    removed_items = result.get("removed_items", [])
    detected_inventory = result.get("detected_inventory", [])
    message = result.get("message", "")
    
    if not shopping_list:
        formatted = []
        formatted.append("# 🛒 Smart Shopping List")
        formatted.append("")
        
        if removed_items:
            formatted.append("## ⚠️ All Items Already Available!")
            formatted.append("According to your inventory analysis, you already have all the ingredients needed for your meal plan.")
            formatted.append("")
            
            if detected_inventory:
                formatted.append("### 🏠 Items Found in Your Inventory:")
                for item in detected_inventory[:10]:  # Show first 10 items
                    formatted.append(f"- {item.title()}")
                if len(detected_inventory) > 10:
                    formatted.append(f"- ... and {len(detected_inventory) - 10} more items")
                formatted.append("")
            
            formatted.append("### 📋 Items That Were Removed:")
            for item in removed_items[:10]:  # Show first 10 removed items
                formatted.append(f"- ~~{item['item']}~~ ({item['quantity']}) - *found in inventory*")
            if len(removed_items) > 10:
                formatted.append(f"- ... and {len(removed_items) - 10} more items")
            formatted.append("")
            
            formatted.append("### 💡 Suggestions:")
            formatted.append("- Double-check your pantry/fridge to make sure you have everything")
            formatted.append("- If this seems incorrect, try re-uploading clearer inventory photos")
            formatted.append("- You can also generate a 'Basic List' to see all ingredients")
        else:
            formatted.append("## 🎉 No Shopping Needed!")
            formatted.append("Your meal plan doesn't require any additional ingredients!")
        
        return "\n".join(formatted)
    
    formatted = []
    formatted.append("# 🛒 Smart Shopping List")
    formatted.append("")
    
    # Add safety message if present
    if message:
        formatted.append(f"## ℹ️ Note: {message}")
        formatted.append("")
    
    # Add filtering summary if applicable
    if removed_items or detected_inventory:
        formatted.append("## 📊 Inventory Analysis Summary")
        if detected_inventory:
            formatted.append(f"**Detected in your inventory:** {', '.join(detected_inventory[:5])}")
            if len(detected_inventory) > 5:
                formatted.append(f" ... and {len(detected_inventory) - 5} more items")
        if removed_items:
            formatted.append(f"**Items removed from list:** {len(removed_items)} items already in your pantry")
        formatted.append("")
        formatted.append("---")
        formatted.append("")
    
    # Display the filtered shopping list
    total_items = 0
    for category in shopping_list:
        category_name = category["category"]
        items = category["items"]
        
        if not items:
            continue
            
        total_items += len(items)
        
        # Add category header
        formatted.append(f"## 📂 {category_name}")
        formatted.append("")
        
        # Add items to buy
        for item in items:
            formatted.append(f"- [ ] **{item['item']}** - {item['quantity']}")
        
        formatted.append("")
        formatted.append("---")
        formatted.append("")
    
    # Add summary
    summary = f"**Total items to buy:** {total_items}"
    if removed_items:
        summary += f" | **Items saved:** {len(removed_items)}"
    
    formatted.insert(2, summary)
    formatted.insert(3, "")
    
    return "\n".join(formatted)

def regenerate_day(day_number: int) -> Tuple[str, str, Union[dict, bool]]:
    """Regenerate meals for a specific day."""
    try:
        if not meal_plan_state["current_plan"] or not meal_plan_state["original_preferences"]:
            return "❌ Error: No meal plan available to regenerate", "", gr.update(visible=False)
        
        # Find the day to regenerate
        day_to_regenerate = None
        for day_data in meal_plan_state["current_plan"]:
            if day_data["day"] == day_number:
                day_to_regenerate = day_data
                break
        
        if not day_to_regenerate:
            return f"❌ Error: Day {day_number} not found in current plan", "", gr.update(visible=False)
        
        # Reset confirmation status since plan is being modified
        meal_plan_state["plan_confirmed"] = False
        meal_plan_state["plan_generation_status"] = "ready"
        meal_plan_state["current_shopping_list"] = None
        
        # Prepare regeneration request
        regeneration_data = {
            "day": day_number,
            "current_meals": {
                "breakfast": day_to_regenerate["breakfast"],
                "lunch": day_to_regenerate["lunch"],
                "dinner": day_to_regenerate["dinner"]
            }
        }
        
        # Regenerate the day
        result = meal_planner_service.regenerate_day(
            meal_plan_state["original_preferences"],
            regeneration_data
        )
        
        if not result["success"]:
            return f"❌ Error: {result['error']}", "", gr.update(visible=False)
        
        # Update the meal plan state
        for i, day_data in enumerate(meal_plan_state["current_plan"]):
            if day_data["day"] == day_number:
                meal_plan_state["current_plan"][i]["breakfast"] = result["breakfast"]
                meal_plan_state["current_plan"][i]["lunch"] = result["lunch"]
                meal_plan_state["current_plan"][i]["dinner"] = result["dinner"]
                break
        
        # Format updated plan for display
        formatted_plan = format_meal_plan_display(meal_plan_state["current_plan"])
        
        return f"✅ Day {day_number} regenerated successfully! Please confirm the updated plan.", formatted_plan, gr.update(visible=True)
        
    except Exception as e:
        logger.error(f"Error regenerating day: {e}")
        return f"❌ Error: {str(e)}", "", gr.update(visible=False)

# Create the Gradio interface
with gr.Blocks(title="Smart Fridge & Meal Planner") as iface:
    gr.Markdown("# 🥘 Smart Fridge Shopping List Generator & AI Meal Planner")
    
    with gr.Tabs():
        # Workflow Overview Tab
        with gr.TabItem("🏠 Smart Shopping Workflow"):
            gr.Markdown("# 🎯 Complete Smart Shopping Workflow")
            gr.Markdown("Generate meal plans and create intelligent shopping lists that automatically filter items you already have!")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## 📋 Workflow Steps")
                    gr.Markdown("""
                    ### 1. 🍽️ Create Your Meal Plan
                    - Go to the **AI Meal Planner** tab
                    - Set your preferences (duration, cuisines, dietary restrictions)
                    - Generate and review your personalized meal plan
                    - Customize individual days if needed
                    - **Confirm your plan** when satisfied
                    
                    ### 2. 🛒 Generate Smart Shopping List
                    - After confirming your meal plan, choose your shopping list type:
                      - **Basic List**: All ingredients from your meal plan
                      - **Smart List**: Upload pantry/fridge photos to automatically remove items you already have
                    
                    ### 3. 📱 Shop Smart
                    - View your categorized shopping list in the **Smart Shopping List** tab
                    - Items you already have are automatically filtered out
                    - Generate QR codes for mobile shopping
                    - Share with family members
                    """)
                
                with gr.Column():
                    gr.Markdown("## ✨ Key Features")
                    gr.Markdown("""
                    ### 🧠 AI-Powered Meal Planning
                    - Personalized meal plans (1-30 days)
                    - Multiple cuisine support
                    - Dietary restriction handling
                    - Individual day customization
                    
                    ### 📷 Smart Inventory Detection
                    - Upload multiple pantry/fridge photos
                    - Automatic food item recognition
                    - Intelligent ingredient matching
                    - Reduces food waste and costs
                    
                    ### 🛒 Intelligent Shopping Lists
                    - Categorized by grocery store sections
                    - Visual status indicators
                    - Mobile-friendly QR codes
                    - Family sharing capabilities
                    """)
            
            with gr.Row():
                gr.Markdown("""
                ## 🚀 Getting Started
                
                **Ready to start?** Click on the **AI Meal Planner** tab above to begin creating your personalized meal plan!
                
                ---
                
                💡 **Pro Tip**: Take photos of your pantry, fridge, and storage areas before generating your shopping list for maximum efficiency!
                """)
            
            # Quick start button
            with gr.Row():
                quick_start_btn = gr.Button("🎯 Start with Meal Planning", variant="primary", size="lg")
            
            # Quick start functionality (just changes tab)
            quick_start_btn.click(
                fn=lambda: gr.update(),  # No-op function
                inputs=[],
                outputs=[]
            )
        
        # Meal Planner Tab
        with gr.TabItem("🍽️ AI Meal Planner"):
            gr.Markdown("Generate personalized meal plans using AI")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Meal Plan Configuration")
                    
                    duration_input = gr.Number(
                        label="Plan Duration (days)",
                        value=7,
                        minimum=1,
                        maximum=30,
                        step=1
                    )
                    
                    cuisine_prefs_input = gr.CheckboxGroup(
                        label="Cuisine Preferences",
                        choices=[
                            "Italian", "Mexican", "Indian", "Japanese", 
                            "Mediterranean", "Chinese", "Thai", "French",
                            "American", "Greek", "Korean", "Any"
                        ],
                        value=["Any"]
                    )
                    
                    dietary_restrictions_input = gr.CheckboxGroup(
                        label="Dietary Restrictions",
                        choices=[
                            "Vegetarian", "Vegan", "Gluten-Free", 
                            "Dairy-Free", "Keto", "Paleo", "Low-Carb", "None"
                        ],
                        value=["None"]
                    )
                    
                    notes_input = gr.Textbox(
                        label="Additional Notes (Optional)",
                        placeholder="e.g., allergic to peanuts, no spicy food...",
                        lines=3
                    )
                    
                    generate_plan_btn = gr.Button("🎯 Generate My Plan", variant="primary", size="lg")
                
                with gr.Column():
                    gr.Markdown("### Generated Meal Plan")
                    plan_status = gr.Textbox(label="Status", lines=1)
                    meal_plan_output = gr.Markdown(label="Meal Plan", value="No meal plan generated yet.")
            
            # Plan confirmation and customization section
            with gr.Row():
                with gr.Column():                    
                    gr.Markdown("### Plan Confirmation")
                    with gr.Row():
                        confirm_plan_btn = gr.Button("✅ Confirm Plan", variant="primary", visible=False)
                    
                    # Day regeneration section
                    gr.Markdown("### Customize Individual Days")
                    day_to_regenerate = gr.Number(
                        label="Day Number to Regenerate",
                        value=1,
                        minimum=1,
                        maximum=30,
                        step=1
                    )
                    regenerate_btn = gr.Button("🔄 Regenerate Day", variant="secondary")
                
                with gr.Column():
                    gr.Markdown("### Create Smart Shopping List")
                    shopping_list_status = gr.Textbox(label="Shopping List Status", lines=1)
                    
                    # Inventory image upload section
                    gr.Markdown("#### 📷 Upload Inventory Images (Optional)")
                    gr.Markdown("*Upload photos of your pantry, fridge, or storage areas to automatically remove items you already have*")
                    
                    with gr.Row():
                        inventory_image1 = gr.Image(
                            type="pil", 
                            label="Pantry/Fridge Image 1", 
                            height=200,
                            width=200
                        )
                        inventory_image2 = gr.Image(
                            type="pil", 
                            label="Pantry/Fridge Image 2", 
                            height=200,
                            width=200
                        )
                    with gr.Row():
                        create_basic_list_btn = gr.Button("📝 Generate Basic List", variant="secondary", visible=False)
                        create_smart_list_btn = gr.Button("🧠 Generate Smart List", variant="primary", size="lg", visible=False)
                    
                    # Shopping list preview
                    shopping_list_preview = gr.Markdown(
                        label="Shopping List Preview",
                        value="Generate and confirm a meal plan to see shopping list preview.",
                        visible=False
                    )
            
            # Event handlers
            generate_plan_btn.click(
                fn=generate_meal_plan,
                inputs=[duration_input, cuisine_prefs_input, dietary_restrictions_input, notes_input],
                outputs=[plan_status, meal_plan_output, confirm_plan_btn]
            )
            
            confirm_plan_btn.click(
                fn=confirm_meal_plan,
                inputs=[],
                outputs=[plan_status, create_basic_list_btn]
            ).then(
                fn=lambda: gr.update(visible=True),
                inputs=[],
                outputs=[create_smart_list_btn]
            )
            
            regenerate_btn.click(
                fn=regenerate_day,
                inputs=[day_to_regenerate],
                outputs=[plan_status, meal_plan_output, confirm_plan_btn]
            )
            
            # Basic shopping list (without inventory filtering)
            create_basic_list_btn.click(
                fn=generate_shopping_list_from_meal_plan,
                inputs=[],
                outputs=[shopping_list_status, shopping_list_preview]
            ).then(
                fn=lambda status, content: gr.update(visible=True),
                inputs=[shopping_list_status, shopping_list_preview],
                outputs=[shopping_list_preview]
            )
            
            # Smart shopping list (with inventory filtering)
            create_smart_list_btn.click(
                fn=lambda img1, img2: generate_smart_shopping_list_with_inventory([img for img in [img1, img2] if img is not None]),
                inputs=[inventory_image1, inventory_image2],
                outputs=[shopping_list_status, shopping_list_preview]
            ).then(
                fn=lambda status, content: gr.update(visible=True),
                inputs=[shopping_list_status, shopping_list_preview],
                outputs=[shopping_list_preview]
            )
        
        # Smart Shopping List Tab
        with gr.TabItem("🛒 Smart Shopping List"):
            gr.Markdown("Your personalized shopping list based on your confirmed meal plan")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Shopping List Management")
                    
                    # Shopping list status and summary
                    with gr.Row():
                        shopping_list_summary = gr.Textbox(
                            label="Shopping List Summary",
                            value="No shopping list available. Generate and confirm a meal plan first.",
                            lines=2,
                            interactive=False
                        )
                    
                    # Action buttons
                    with gr.Row():
                        refresh_shopping_list_btn = gr.Button("🔄 Refresh List", variant="secondary")
                        filter_pantry_btn = gr.Button("🏠 Re-scan Pantry", variant="secondary")
                        export_shopping_list_btn = gr.Button("📱 Generate QR Code", variant="primary")
                
                with gr.Column():
                    gr.Markdown("### QR Code Export")
                    shopping_list_qr = gr.Image(label="Shopping List QR Code")
                    qr_instructions = gr.Markdown(
                        "**QR Code Instructions:**\n"
                        "- Scan with your phone to get the shopping list\n"
                        "- Only items you need to buy are included\n"
                        "- Items already in your pantry are filtered out\n"
                        "- Share with family members for coordinated shopping"
                    )
            
            with gr.Row():
                shopping_list_display = gr.Markdown(
                    label="Detailed Shopping List",
                    value="No shopping list available. Please generate a meal plan first and create a shopping list."
                )
            
            # Enhanced shopping list event handlers
            def get_shopping_list_summary():
                """Get a summary of the current shopping list."""
                try:
                    if not meal_plan_state.get("current_shopping_list"):
                        return "No shopping list available. Generate and confirm a meal plan first."
                    
                    shopping_list = meal_plan_state["current_shopping_list"]
                    total_items = 0
                    categories = 0
                    
                    for category in shopping_list:
                        if category["items"]:
                            categories += 1
                            for item in category["items"]:
                                total_items += 1
                    
                    # Check if we have removed items info (from smart shopping list)
                    removed_items_count = 0
                    if hasattr(meal_plan_state, 'last_removed_items'):
                        removed_items_count = len(meal_plan_state.get('last_removed_items', []))
                    
                    if removed_items_count > 0:
                        return (f"📊 **Summary:** {categories} categories, {total_items} items to buy\n"
                               f"🛒 {total_items} items needed | 🏠 {removed_items_count} items already in pantry")
                    else:
                        return (f"📊 **Summary:** {categories} categories, {total_items} total items\n"
                               f"🛒 {total_items} items to buy")
                    
                except Exception as e:
                    logger.error(f"Error getting shopping list summary: {e}")
                    return "Error retrieving shopping list summary."
            
            def refresh_shopping_list():
                """Refresh the shopping list display and summary."""
                try:
                    summary = get_shopping_list_summary()
                    
                    if meal_plan_state.get("current_shopping_list"):
                        display = format_shopping_list_display(meal_plan_state["current_shopping_list"])
                    else:
                        display = "No shopping list available. Please generate a meal plan first and create a shopping list."
                    
                    return summary, display
                    
                except Exception as e:
                    logger.error(f"Error refreshing shopping list: {e}")
                    return "Error refreshing shopping list.", "Error loading shopping list."
            
            def rescan_pantry():
                """Re-scan pantry and update shopping list filtering."""
                try:
                    if not meal_plan_state.get("current_plan"):
                        return "No meal plan available to regenerate shopping list from.", "No shopping list available."
                    
                    # Regenerate shopping list with fresh pantry scan
                    result = meal_planner_service.generate_shopping_list(meal_plan_state["current_plan"])
                    
                    if result["success"]:
                        meal_plan_state["current_shopping_list"] = result["shoppingList"]
                        summary = get_shopping_list_summary()
                        display = format_shopping_list_display(result["shoppingList"])
                        return summary, display
                    else:
                        return f"Error re-scanning pantry: {result['error']}", "Error updating shopping list."
                        
                except Exception as e:
                    logger.error(f"Error re-scanning pantry: {e}")
                    return "Error re-scanning pantry.", "Error updating shopping list."
            
            refresh_shopping_list_btn.click(
                fn=refresh_shopping_list,
                inputs=[],
                outputs=[shopping_list_summary, shopping_list_display]
            )
            
            filter_pantry_btn.click(
                fn=rescan_pantry,
                inputs=[],
                outputs=[shopping_list_summary, shopping_list_display]
            )
            
            def generate_shopping_list_qr():
                """Generate QR code for the current shopping list."""
                try:
                    if not meal_plan_state.get("current_shopping_list"):
                        return None
                    
                    # Convert shopping list to simple format for QR code
                    simple_list = []
                    for category in meal_plan_state["current_shopping_list"]:
                        for item in category["items"]:
                            # In the SmartShoppingList module, items don't have status field
                            # All items in the list are items to buy (filtered items are removed)
                            simple_list.append({
                                "name": item["item"],
                                "quantity": item["quantity"],
                                "category": category["category"]
                            })
                    
                    if not simple_list:
                        logger.warning("No items needed for shopping list QR code")
                        return None
                    
                    return generate_qr_code(simple_list)
                    
                except Exception as e:
                    logger.error(f"Error generating shopping list QR: {e}")
                    return None
            
            export_shopping_list_btn.click(
                fn=generate_shopping_list_qr,
                inputs=[],
                outputs=[shopping_list_qr]
            )
            
            # Auto-refresh summary when tab is loaded
            refresh_shopping_list_btn.click(
                fn=get_shopping_list_summary,
                inputs=[],
                outputs=[shopping_list_summary]
            )
        
        # Instructions
        with gr.TabItem("📖 How to Use"):
            gr.Markdown("""
            ## 🎯 Smart Shopping Workflow (Complete Guide)
            ### Step 1: Create Your Meal Plan
            1. **Navigate to AI Meal Planner tab**
            2. **Configure Your Plan:**
               - Set your desired plan duration (1-30 days)
               - Select your cuisine preferences
               - Choose any dietary restrictions
               - Add any additional notes about allergies or preferences
            
            3. **Generate & Review:**
               - Click "Generate My Plan"
               - Review the generated meal plan
               - Use "Regenerate Day" to customize specific days if needed
            
            4. **Confirm Your Plan:**
               - Click "Confirm Plan" when you're happy with your meals
               - This enables smart shopping list generation
            
            ### Step 2: Generate Smart Shopping List
            5. **Choose Your List Type:**
               - **Basic List**: Click "Generate Basic List" for all ingredients
               - **Smart List**: Upload pantry/fridge photos first, then click "Generate Smart List"
            
            6. **Upload Inventory Images (for Smart List):**
               - Take photos of your pantry, fridge, and storage areas
               - Upload up to 2 images before generating smart list
               - AI will automatically detect items you already have
            
            ### Step 3: Shop Smart
            7. **View Your List in Smart Shopping List tab:**
               - See categorized items (Produce, Pantry Staples, etc.)
               - Items filtered out are shown as "already in pantry"
               - Get a summary of items to buy vs. items you have
            
            8. **Export & Share:**
               - Generate QR codes with only items you need to buy
               - Share with family members for coordinated shopping
               - Use on your phone while shopping
            
            ## ✨ Key Benefits
            - **Saves Money**: Avoid buying items you already have
            - **Reduces Waste**: Use ingredients you already own
            - **Saves Time**: Organized, filtered shopping lists
            - **Family Friendly**: Easy sharing and coordination
            """)

if __name__ == "__main__":
    # Always set share=True for Hugging Face Spaces to create a public link
    iface.launch(share=True, quiet=True)
