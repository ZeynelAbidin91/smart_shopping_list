"""
Configuration settings for the Smart Fridge Shopping List Generator
"""

# Diet restrictions for different diets
DIET_RESTRICTIONS = {
    "vegetarian": ["meat", "fish", "chicken", "beef", "pork"],
    "vegan": ["meat", "fish", "chicken", "beef", "pork", "eggs", "milk", "cheese", "honey"],
    "keto": ["bread", "pasta", "rice", "sugar", "potato", "corn"],
    "gluten-free": ["wheat", "bread", "pasta", "barley", "rye"]
}

# Cuisine-based suggestions
CUISINE_SUGGESTIONS = {
    "Italian": [
        "pasta", "tomatoes", "olive oil", "basil", "mozzarella",
        "parmesan", "garlic", "oregano", "balsamic vinegar"
    ],
    "Indian": [
        "rice", "lentils", "curry powder", "ginger", "garlic",
        "turmeric", "cumin", "coriander", "chickpeas"
    ],
    "Mexican": [
        "tortillas", "beans", "avocado", "cilantro", "lime",
        "jalapeño", "tomatoes", "onions", "cheese"
    ],
    "Japanese": [
        "rice", "nori", "soy sauce", "miso", "tofu",
        "wasabi", "ginger", "sake", "mirin"
    ],
    "Mediterranean": [
        "olive oil", "feta cheese", "hummus", "pita", "cucumber",
        "tomatoes", "olives", "yogurt", "herbs"
    ]
}

# Object detection settings
DETECTION_CONFIDENCE_THRESHOLD = 0.5
MIN_OBJECT_AREA = 1000  # minimum area for contour detection

# UI Settings
MAX_SUGGESTED_ITEMS = 20
SUPPORTED_DIETS = ["None", "Vegetarian", "Vegan", "Keto", "Gluten-free"]
SUPPORTED_CUISINES = list(CUISINE_SUGGESTIONS.keys())

# Notification settings
MAX_SMS_LENGTH = 1600  # Maximum length for SMS messages
