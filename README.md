# Smart Fridge Shopping List Generator

This is a Gradio app that helps users generate shopping lists based on images of their refrigerator contents. The app uses computer vision to detect food items, considers dietary preferences and cuisine familiarity, and generates a QR code for easy transfer to your smartphone.

## Features

- Image-based food item detection using computer vision
- Diet and cuisine preference filtering
- Interactive shopping list customization
- QR code generation for easy smartphone access
- User-friendly Gradio interface

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

## Running the App

1. Activate your virtual environment if not already active
2. Run the app:
   ```bash
   python app.py
   ```
3. Open the provided URL (http://127.0.0.1:7860) in your browser

## Usage

1. Upload a photo of your refrigerator contents
2. Select your dietary preferences
3. Enter cuisines you're familiar with (comma-separated)
4. Click "Process Image" to generate the initial list
5. Adjust quantities and select/deselect items as needed
6. Click "Generate QR Code" to create a QR code
7. Scan the QR code with your smartphone's camera to view the list

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
