import os
import torch

# Set OpenMP environment variable to handle multiple runtime versions
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Check CUDA availability and set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}")

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

# Global state for user preferences
user_preferences = {
    "diet": None,
    "cuisines": []
}

# Initialize the shopping list agent and meal planner
shopping_list_agent = ShoppingListAgent()
meal_planner_service = MealPlannerService()

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
    """Generate a QR code for the shopping list."""
    try:
        # Convert list to JSON
        list_data = json.dumps([{
            "name": item["name"],
            "quantity": item["quantity"],
            "to_purchase": item.get("to_purchase", True)
        } for item in items])
        
        # Create QR code
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(list_data)
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
def generate_meal_plan(duration: int, cuisine_prefs: str, dietary_restrictions: str, notes: str) -> Tuple[str, str, str]:
    """Generate a meal plan based on user preferences."""
    try:
        # Reset state
        meal_plan_state["plan_confirmed"] = False
        meal_plan_state["plan_generation_status"] = "generating"
        meal_plan_state["current_shopping_list"] = None
        
        # Validate duration
        if not duration or duration < 1:
            meal_plan_state["plan_generation_status"] = "not_started"
            return "❌ Error: Duration must be at least 1 day", "", "hidden"
        
        # Parse preferences
        cuisine_list = [c.strip() for c in cuisine_prefs.split(",") if c.strip()] if cuisine_prefs else []
        dietary_list = [d.strip() for d in dietary_restrictions.split(",") if d.strip()] if dietary_restrictions else []
        
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
            return f"❌ Error: {result['error']}", "", "hidden"
        
        # Store the meal plan
        meal_plan_state["current_plan"] = result["mealPlan"]
        meal_plan_state["plan_generation_status"] = "ready"
        
        # Format for display
        formatted_plan = format_meal_plan_display(result["mealPlan"])
        
        return "✅ Meal plan generated successfully! Review and confirm to create shopping list.", formatted_plan, "visible"
        
    except Exception as e:
        logger.error(f"Error generating meal plan: {e}")
        meal_plan_state["plan_generation_status"] = "not_started"
        return f"❌ Error: {str(e)}", "", "hidden"

def confirm_meal_plan() -> Tuple[str, str]:
    """Confirm the current meal plan and enable shopping list generation."""
    try:
        if meal_plan_state["plan_generation_status"] != "ready":
            return "❌ Error: No meal plan available to confirm. Please generate a meal plan first.", "hidden"
        
        if not meal_plan_state["current_plan"]:
            return "❌ Error: No meal plan data found.", "hidden"
        
        # Mark plan as confirmed
        meal_plan_state["plan_confirmed"] = True
        meal_plan_state["plan_generation_status"] = "confirmed"
        
        return "✅ Meal plan confirmed! You can now generate your shopping list.", "visible"
        
    except Exception as e:
        logger.error(f"Error confirming meal plan: {e}")
        return f"❌ Error: {str(e)}", "hidden"

def generate_shopping_list_from_meal_plan() -> Tuple[str, str, str]:
    """Generate a shopping list from the confirmed meal plan."""
    try:
        if not meal_plan_state["plan_confirmed"]:
            return "❌ Error: Please confirm your meal plan first before generating a shopping list.", "", "hidden"
        
        if not meal_plan_state["current_plan"]:
            return "❌ Error: No meal plan available. Please generate and confirm a meal plan first.", "", "hidden"
        
        # Generate shopping list
        result = meal_planner_service.generate_shopping_list(meal_plan_state["current_plan"])
        
        if not result["success"]:
            return f"❌ Error: {result['error']}", "", "hidden"
        
        # Store the shopping list
        meal_plan_state["current_shopping_list"] = result["shoppingList"]
        
        # Format for display
        formatted_shopping_list = format_shopping_list_display(result["shoppingList"])
        
        return "✅ Shopping list generated successfully! Check the Smart Shopping List tab.", formatted_shopping_list, "visible"
        
    except Exception as e:
        logger.error(f"Error generating shopping list: {e}")
        return f"❌ Error: {str(e)}", "", "hidden"

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
        needed_items_in_category = [item for item in items if item["status"] == "needed"]
        pantry_items_in_category = [item for item in items if item["status"] == "in_pantry"]
        
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
    summary = f"**Summary:** {needed_items_count} items to buy, {pantry_items_count} items already in pantry"
    formatted.insert(2, summary)
    formatted.insert(3, "")
    
    return "\n".join(formatted)

def regenerate_day(day_number: int) -> Tuple[str, str, str]:
    """Regenerate meals for a specific day."""
    try:
        if not meal_plan_state["current_plan"] or not meal_plan_state["original_preferences"]:
            return "❌ Error: No meal plan available to regenerate", "", "hidden"
        
        # Find the day to regenerate
        day_to_regenerate = None
        for day_data in meal_plan_state["current_plan"]:
            if day_data["day"] == day_number:
                day_to_regenerate = day_data
                break
        
        if not day_to_regenerate:
            return f"❌ Error: Day {day_number} not found in current plan", "", "hidden"
        
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
            return f"❌ Error: {result['error']}", "", "hidden"
        
        # Update the meal plan state
        for i, day_data in enumerate(meal_plan_state["current_plan"]):
            if day_data["day"] == day_number:
                meal_plan_state["current_plan"][i]["breakfast"] = result["breakfast"]
                meal_plan_state["current_plan"][i]["lunch"] = result["lunch"]
                meal_plan_state["current_plan"][i]["dinner"] = result["dinner"]
                break
        
        # Format updated plan for display
        formatted_plan = format_meal_plan_display(meal_plan_state["current_plan"])
        
        return f"✅ Day {day_number} regenerated successfully! Please confirm the updated plan.", formatted_plan, "visible"
        
    except Exception as e:
        logger.error(f"Error regenerating day: {e}")
        return f"❌ Error: {str(e)}", "", "hidden"

# Create the Gradio interface
with gr.Blocks(title="Smart Fridge & Meal Planner") as iface:
    gr.Markdown("# 🥘 Smart Fridge Shopping List Generator & AI Meal Planner")
    
    with gr.Tabs():
        # Shopping List Tab
        with gr.TabItem("📱 Shopping List Generator"):
            gr.Markdown("Upload an image of your fridge or pantry to get a smart shopping list")
            
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="pil", label="Upload Fridge/Pantry Image")
                    diet_input = gr.Dropdown(
                        choices=["None", "Vegetarian", "Vegan", "Keto", "Gluten-free"],
                        label="Dietary Preference",
                        value="None"
                    )
                    cuisine_input = gr.Textbox(
                        label="Preferred Cuisines (comma-separated)",
                        placeholder="Italian, Mexican, Indian..."
                    )
                    submit_btn = gr.Button("Generate Shopping List", variant="primary")
                
                with gr.Column():
                    qr_output = gr.Image(label="Shopping List QR Code")
                    list_output = gr.Textbox(label="Shopping List", lines=10)
            
            submit_btn.click(
                fn=process_image_and_generate_list,
                inputs=[image_input, diet_input, cuisine_input],
                outputs=[qr_output, list_output]
            )        # Meal Planner Tab
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
                        confirm_plan_btn = gr.Button("✅ Confirm Plan", variant="primary")
                        confirm_visibility = gr.State("hidden")  # Hidden by default
                    
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
                    gr.Markdown("### Create Shopping List")
                    shopping_list_status = gr.Textbox(label="Shopping List Status", lines=1)
                    create_shopping_list_btn = gr.Button("🛒 Create Shopping List", variant="primary", size="lg")
                    shopping_list_visibility = gr.State("hidden")  # Hidden by default
                    
                    # Shopping list preview
                    shopping_list_preview = gr.Markdown(
                        label="Shopping List Preview",
                        value="Generate and confirm a meal plan to see shopping list preview.",
                        visible=False
                    )
            
            # Event handlers
            generate_plan_btn.click(
                fn=lambda d, c, dr, n: generate_meal_plan(d, ",".join(c), ",".join(dr), n),
                inputs=[duration_input, cuisine_prefs_input, dietary_restrictions_input, notes_input],
                outputs=[plan_status, meal_plan_output, confirm_visibility]
            ).then(
                fn=lambda visibility: gr.update(visible=(visibility == "visible")),
                inputs=[confirm_visibility],
                outputs=[confirm_plan_btn]
            )
            
            confirm_plan_btn.click(
                fn=confirm_meal_plan,
                inputs=[],
                outputs=[plan_status, shopping_list_visibility]
            ).then(
                fn=lambda visibility: gr.update(visible=(visibility == "visible")),
                inputs=[shopping_list_visibility],
                outputs=[create_shopping_list_btn]
            )
            
            regenerate_btn.click(
                fn=regenerate_day,
                inputs=[day_to_regenerate],
                outputs=[plan_status, meal_plan_output, confirm_visibility]
            ).then(
                fn=lambda visibility: gr.update(visible=(visibility == "visible")),
                inputs=[confirm_visibility],
                outputs=[confirm_plan_btn]
            )
            
            create_shopping_list_btn.click(
                fn=generate_shopping_list_from_meal_plan,
                inputs=[],
                outputs=[shopping_list_status, shopping_list_preview, shopping_list_visibility]
            ).then(
                fn=lambda visibility: gr.update(visible=(visibility == "visible")),
                inputs=[shopping_list_visibility],
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
                    needed_items = 0
                    pantry_items = 0
                    categories = 0
                    
                    for category in shopping_list:
                        if category["items"]:
                            categories += 1
                            for item in category["items"]:
                                total_items += 1
                                if item["status"] == "needed":
                                    needed_items += 1
                                else:
                                    pantry_items += 1
                    
                    return (f"📊 **Summary:** {categories} categories, {total_items} total items\n"
                           f"🛒 {needed_items} items to buy | 🏠 {pantry_items} items in pantry")
                    
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
                            if item["status"] == "needed":  # Only include items to buy
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
            )    # Instructions
    with gr.Accordion("📖 How to Use", open=False):
        gr.Markdown("""
        ## 📱 Shopping List Generator
        1. Upload a photo of your fridge or pantry
        2. Select your dietary preference (if any)
        3. Enter your preferred cuisines (optional)
        4. Click "Generate Shopping List"
        5. Get your shopping list and QR code!
        
        ## 🍽️ AI Meal Planner (Complete Workflow)
        1. **Configure Your Plan:**
           - Set your desired plan duration (1-30 days)
           - Select your cuisine preferences
           - Choose any dietary restrictions
           - Add any additional notes about allergies or preferences
        
        2. **Generate & Review:**
           - Click "Generate My Plan"
           - Review the generated meal plan
           - Use "Regenerate Day" to customize specific days if needed
        
        3. **Confirm Your Plan:**
           - Click "Confirm Plan" when you're happy with your meals
           - This enables shopping list generation
        
        4. **Create Shopping List:**
           - Click "Create Shopping List" after confirming your plan
           - View the detailed shopping list in the Smart Shopping List tab
        
        ## 🛒 Smart Shopping List (Enhanced Features)
        1. **View Your List:**
           - See categorized items (Produce, Pantry Staples, etc.)
           - Items you likely already have are marked as "in pantry" and crossed out
           - Get a summary of items to buy vs. items already available
        
        2. **Manage Your List:**
           - Click "Refresh List" to update the display
           - Use "Re-scan Pantry" to re-analyze your pantry contents
           - Items are automatically filtered based on what you likely have
        
        3. **Export & Share:**
           - Generate QR codes with only items you need to buy
           - Share with family members for coordinated shopping
           - Use on your phone while shopping
        
        ## ✨ Key Features
        - **Smart Pantry Detection**: AI analyzes your fridge/pantry contents
        - **Personalized Meal Plans**: AI-generated plans tailored to your preferences
        - **Confirmation Workflow**: Review and approve plans before shopping
        - **Day-by-Day Customization**: Don't like a specific day? Regenerate it!
        - **Intelligent Filtering**: Automatically removes items you likely have
        - **Categorized Lists**: Organized shopping by grocery store sections
        - **QR Code Export**: Easy mobile access to your shopping list
        - **Family Sharing**: Share lists with household members
        - **Dietary Support**: Full support for dietary restrictions and preferences
        
        ## 🔄 Workflow Summary
        **Generate Plan → Review & Customize → Confirm → Create Shopping List → Shop Smart**
        
        Each step ensures you get exactly what you want while minimizing food waste and shopping time!
        """)

if __name__ == "__main__":
    iface.launch()
