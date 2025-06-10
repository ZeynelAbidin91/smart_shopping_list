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
    "original_preferences": None
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
def generate_meal_plan(duration: int, cuisine_prefs: str, dietary_restrictions: str, notes: str) -> Tuple[str, str]:
    """Generate a meal plan based on user preferences."""
    try:
        # Validate duration
        if not duration or duration < 1:
            return "❌ Error: Duration must be at least 1 day", ""
        
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
            return f"❌ Error: {result['error']}", ""
        
        # Store the meal plan
        meal_plan_state["current_plan"] = result["mealPlan"]
        
        # Format for display
        formatted_plan = format_meal_plan_display(result["mealPlan"])
        
        return "✅ Meal plan generated successfully!", formatted_plan
        
    except Exception as e:
        logger.error(f"Error generating meal plan: {e}")
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

def regenerate_day(day_number: int) -> Tuple[str, str]:
    """Regenerate meals for a specific day."""
    try:
        if not meal_plan_state["current_plan"] or not meal_plan_state["original_preferences"]:
            return "❌ Error: No meal plan available to regenerate", ""
        
        # Find the day to regenerate
        day_to_regenerate = None
        for day_data in meal_plan_state["current_plan"]:
            if day_data["day"] == day_number:
                day_to_regenerate = day_data
                break
        
        if not day_to_regenerate:
            return f"❌ Error: Day {day_number} not found in current plan", ""
        
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
            return f"❌ Error: {result['error']}", ""
        
        # Update the meal plan state
        for i, day_data in enumerate(meal_plan_state["current_plan"]):
            if day_data["day"] == day_number:
                meal_plan_state["current_plan"][i]["breakfast"] = result["breakfast"]
                meal_plan_state["current_plan"][i]["lunch"] = result["lunch"]
                meal_plan_state["current_plan"][i]["dinner"] = result["dinner"]
                break
        
        # Format updated plan for display
        formatted_plan = format_meal_plan_display(meal_plan_state["current_plan"])
        
        return f"✅ Day {day_number} regenerated successfully!", formatted_plan
        
    except Exception as e:
        logger.error(f"Error regenerating day: {e}")
        return f"❌ Error: {str(e)}", ""

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
            
            # Day regeneration section
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Regenerate Specific Day")
                    day_to_regenerate = gr.Number(
                        label="Day Number to Regenerate",
                        value=1,
                        minimum=1,
                        maximum=30,
                        step=1
                    )
                    regenerate_btn = gr.Button("🔄 Regenerate Day", variant="secondary")
            
            # Event handlers
            generate_plan_btn.click(
                fn=lambda d, c, dr, n: generate_meal_plan(d, ",".join(c), ",".join(dr), n),
                inputs=[duration_input, cuisine_prefs_input, dietary_restrictions_input, notes_input],
                outputs=[plan_status, meal_plan_output]
            )
            
            regenerate_btn.click(
                fn=regenerate_day,
                inputs=[day_to_regenerate],
                outputs=[plan_status, meal_plan_output]
            )
    
    # Instructions
    with gr.Accordion("📖 How to Use", open=False):
        gr.Markdown("""
        ## Shopping List Generator
        1. Upload a photo of your fridge or pantry
        2. Select your dietary preference (if any)
        3. Enter your preferred cuisines (optional)
        4. Click "Generate Shopping List"
        5. Get your shopping list and QR code!
        
        ## AI Meal Planner
        1. Set your desired plan duration (1-30 days)
        2. Select your cuisine preferences
        3. Choose any dietary restrictions
        4. Add any additional notes about allergies or preferences
        5. Click "Generate My Plan"
        6. Use "Regenerate Day" to get new options for specific days
        
        ## Features
        - **Smart Detection**: AI analyzes your fridge contents
        - **Personalized Plans**: Meal plans tailored to your preferences
        - **Dietary Support**: Vegetarian, vegan, keto, gluten-free options
        - **QR Codes**: Easy sharing of shopping lists
        - **Day Regeneration**: Don't like a day? Regenerate it!
        """)

if __name__ == "__main__":
    iface.launch()
