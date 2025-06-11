---
title: Smart Shopping VLM
emoji: 🛒
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: mit
tags:
  - computer-vision
  - vision-language-model
  - shopping
  - meal-planning
  - qwen2.5-vl  - food-detection
  - agent-demo-track
short_description: AI-powered smart shopping lists with meal planning
models:
  - Qwen/Qwen2.5-VL-7B-Instruct
suggested_hardware: t4-small
suggested_storage: small
---

# Smart Fridge Shopping List Generator & AI Meal Planner

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/your-username/smart-shopping-vlm)

This is a comprehensive AI-powered application that helps users generate smart shopping lists based on images of their refrigerator contents and create personalized meal plans. The app uses advanced computer vision with Qwen2.5 VLM for image analysis and OpenAI's GPT-4o mini for intelligent meal planning.

## 🚀 Live Demo

Try the app now: **[Launch on Hugging Face Spaces](https://huggingface.co/spaces/your-username/smart-shopping-vlm)**

## Features

### 📱 Shopping List Generator
- Advanced image-based food detection using Qwen2.5 VLM
- Intelligent list processing with LlamaIndex agents
- Diet and cuisine preference filtering
- Smart complementary item suggestions
- QR code generation for easy smartphone access

### 🍽️ AI Meal Planner
- Personalized meal plan generation using OpenAI GPT-4o mini
- Support for 1-30 day meal plans
- Multiple cuisine preferences (Italian, Mexican, Indian, Japanese, etc.)
- Dietary restriction support (Vegetarian, Vegan, Keto, Gluten-Free, etc.)
- Per-day meal regeneration for customization
- Detailed meal descriptions and instructions

### 🛒 Smart Shopping List Generator
- **From Meal Plans**: Generate comprehensive shopping lists from confirmed meal plans
- **Smart Filtering**: Automatically detects items you likely already have in your pantry
- **Categorized Lists**: Organized by categories (Produce, Pantry Staples, Protein, etc.)
- **Status Indicators**: Visual distinction between items to buy vs. items in pantry
- **QR Code Export**: Generate QR codes containing only items you need to purchase

## 🏗️ Architecture

The app uses an intelligent multi-agent system:
- **🔍 Image Analysis Agent**: GPU-optimized Qwen2.5 VLM for accurate food item detection
- **🛒 Shopping List Agent**: Handles dietary preferences and smart suggestions  
- **🍽️ Meal Planning Service**: Uses OpenAI GPT-4o mini for personalized meal plans
- **🧠 Smart Filtering**: Automatically removes items you already have from shopping lists

## 🛠️ Setup

### Local Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/smart-shopping-vlm.git
   cd smart-shopping-vlm
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   ```

### Hugging Face Spaces Deployment

This app is deployed on Hugging Face Spaces with GPU support for optimal performance. The deployment automatically handles:
- ✅ GPU memory optimization for RTX A4000/A10G instances
- ✅ Automatic fallback to CPU if GPU is unavailable  
- ✅ Intelligent model caching and loading strategies

## API Keys Required

- **OpenAI API Key**: Required for the AI Meal Planner feature
  - Get your key from: https://platform.openai.com/api-keys
  - Add to `.env` file as: `OPENAI_API_KEY=your_key_here`

## 🎯 Quick Start

### Using Hugging Face Spaces (Recommended)
1. Visit the [live demo](https://huggingface.co/spaces/your-username/smart-shopping-vlm)
2. Upload your fridge/pantry photos
3. Generate meal plans and smart shopping lists instantly!

### Local Development
1. **Test the setup:**
   ```bash
   python test.py
   ```

2. **Run the application:**
   ```bash
   python run.py
   # Or on Windows: powershell -ExecutionPolicy Bypass .\start.ps1
   ```

3. Open your browser to `http://localhost:7860`

## Usage

### Shopping List Generator
1. Navigate to the "Shopping List Generator" tab
2. Upload a photo of your refrigerator contents
3. Select your dietary preferences
4. Enter cuisines you're familiar with
5. Click "Generate Shopping List"
6. Get your personalized shopping list and QR code

### AI Meal Planner
1. Navigate to the "AI Meal Planner" tab
2. Set your desired plan duration (1-30 days)
3. Select cuisine preferences (Italian, Mexican, Indian, etc.)
4. Choose dietary restrictions (Vegetarian, Vegan, Keto, etc.)
5. Add any additional notes (allergies, dislikes)
6. Click "Generate My Plan"
7. Use "Regenerate Day" to customize specific days
8. Click "Create Shopping List" to generate a smart shopping list

### Smart Shopping List
1. After generating a meal plan, click "Create Shopping List"
2. View the categorized shopping list in the "Smart Shopping List" tab
3. Items are automatically sorted into categories (Produce, Pantry Staples, etc.)
4. Items you likely have are marked "in pantry" and crossed out
5. Generate a QR code containing only items you need to buy
6. Use the QR code for convenient mobile shopping

## Testing

Test the meal planner functionality:
```bash
python test_meal_planner.py
```

Test the shopping list generation:
```bash
python test_shopping_list.py
```

Test the complete workflow:
```bash
python test_shopping_list.py  # Includes full workflow test
```

## 🚀 Deployment

### Hugging Face Spaces

This app is optimized for deployment on Hugging Face Spaces with GPU support:

1. **Fork/Clone this repository**
2. **Create a new Space on Hugging Face:**
   - Choose "Gradio" as the SDK
   - Set hardware to "T4 Small" or higher for GPU acceleration
   - Upload files or connect via Git

3. **Configure Secrets:**
   - Add `OPENAI_API_KEY` in your Space settings
   - Set any other required environment variables

4. **Deployment Configuration:**
   - The `README_HF.md` contains the necessary metadata
   - GPU acceleration is automatically detected and utilized
   - Fallback to CPU mode if GPU is unavailable

### Local Development
```bash
python test.py    # Test the setup
python run.py     # Start the application
```

## 🧪 Testing

Test the environment and functionality:
```bash
python test.py
```

This will verify:
- ✅ Python and PyTorch installation
- ✅ CUDA/GPU availability  
- ✅ OpenAI API key configuration
- ✅ Model loading and image analysis
- ✅ Meal planning service

## How It Works

The app processes your fridge image using computer vision to detect food items. It then:
1. Filters items based on your dietary preferences
2. Suggests additional items based on your preferred cuisines
3. Allows you to customize the list
4. Generates a QR code containing your shopping list
5. When scanned, displays a formatted text list on your phone

## 📁 Project Structure

```
smart_shopping_list/
├── 🚀 app.py                 # Main Gradio application
├── 🤖 llama_agents.py        # GPU-optimized image analysis & shopping agents
├── 🍽️ meal_planner.py        # AI meal planning service with OpenAI integration
├── ⚙️ requirements.txt       # Python dependencies
├── 🧪 test.py               # Environment and functionality tests
├── 🏃 run.py                # Simple application launcher
├── 💻 start.ps1             # Windows PowerShell launcher
├── 📖 README.md             # This documentation
├── 🔧 .env.example          # Environment variables template
└── 📱 qr_codes/             # Generated QR codes for mobile access
```

## Dependencies

- **gradio**: Web interface framework
- **torch & transformers**: Deep learning and model execution
- **pillow**: Image processing
- **qrcode**: QR code generation
- **openai**: GPT-4o mini integration for meal planning
- **llama-index**: Agent framework for image analysis
- **loguru**: Enhanced logging

## 🏷️ Tags

This project is part of the **agent-demo-track** showcase, demonstrating:
- Multi-modal AI agents with vision and language capabilities
- Intelligent workflow automation for daily tasks
- GPU-optimized inference with fallback strategies
- Real-world application of computer vision and NLP

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test the complete workflow
5. Submit a pull request

## License

MIT License - feel free to use and modify as needed.
