# Smart Fridge Shopping List Generator & AI Meal Planner

This is a comprehensive Gradio app that helps users generate shopping lists based on images of their refrigerator contents and create personalized meal plans using AI. The app uses LlamaIndex's agent system with Qwen2.5 VLM for image analysis and OpenAI's GPT-4o mini for intelligent meal planning.

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

## Architecture

The app uses a multi-agent system powered by LlamaIndex and OpenAI:
- **Image Analysis Agent**: Uses Qwen2.5 VLM for accurate food item detection
- **Shopping List Agent**: Handles dietary preferences and smart suggestions
- **Meal Planning Service**: Uses OpenAI GPT-4o mini for personalized meal plans

## Setup

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   ```

## API Keys Required

- **OpenAI API Key**: Required for the AI Meal Planner feature
  - Get your key from: https://platform.openai.com/api-keys
  - Add to `.env` file as: `OPENAI_API_KEY=your_key_here`

## Running the App

1. Activate your virtual environment if not already active
2. Run the app:
   ```bash
   python app.py
   ```
3. Open the provided URL (http://127.0.0.1:7860) in your browser

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

## How It Works

The app processes your fridge image using computer vision to detect food items. It then:
1. Filters items based on your dietary preferences
2. Suggests additional items based on your preferred cuisines
3. Allows you to customize the list
4. Generates a QR code containing your shopping list
5. When scanned, displays a formatted text list on your phone

## Project Structure

```
gradio_shop_list_app/
├── app.py              # Main Gradio application
├── config.py           # Configuration settings
├── utils.py            # Utility functions and classes
├── test_utils.py       # Unit tests
└── requirements.txt    # Project dependencies
```

## Dependencies

- gradio: Web interface
- opencv-python: Image processing
- pillow: Image handling
- qrcode: QR code generation
- numpy: Numerical operations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests to ensure everything works
5. Submit a pull request

## License

MIT License - feel free to use and modify as needed.
