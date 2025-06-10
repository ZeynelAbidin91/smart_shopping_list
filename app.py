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

# Initialize the shopping list agent
shopping_list_agent = ShoppingListAgent()

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

# Create the Gradio interface
with gr.Blocks() as iface:
    gr.Markdown("# Smart Fridge Shopping List Generator")
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
            submit_btn = gr.Button("Generate Shopping List")
        
        with gr.Column():
            qr_output = gr.Image(label="Shopping List QR Code")
            list_output = gr.Textbox(label="Shopping List", lines=10)
    
    submit_btn.click(
        fn=process_image_and_generate_list,
        inputs=[image_input, diet_input, cuisine_input],
        outputs=[qr_output, list_output]
    )
    
    gr.Markdown("""
    ## How to Use
    1. Upload a photo of your fridge or pantry
    2. Select your dietary preference (if any)
    3. Enter your preferred cuisines (optional)
    4. Click "Generate Shopping List"
    5. Get your shopping list and QR code!
    """)

if __name__ == "__main__":
    iface.launch()
