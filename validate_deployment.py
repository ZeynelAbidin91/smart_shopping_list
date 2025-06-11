#!/usr/bin/env python3
"""
Validate Hugging Face Spaces deployment configuration.
"""

import os
import yaml
import re

def validate_readme():
    """Validate README.md has proper YAML frontmatter."""
    print("🔍 Validating README.md configuration...")
    
    with open('README.md', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for YAML frontmatter
    if not content.startswith('---'):
        print("❌ README.md missing YAML frontmatter")
        return False
    
    # Extract YAML frontmatter
    try:
        yaml_end = content.find('---', 3)
        if yaml_end == -1:
            print("❌ README.md YAML frontmatter not properly closed")
            return False
        
        yaml_content = content[3:yaml_end].strip()
        config = yaml.safe_load(yaml_content)
        
        # Validate required fields
        required_fields = ['title', 'emoji', 'sdk', 'app_file']
        for field in required_fields:
            if field not in config:
                print(f"❌ Missing required field: {field}")
                return False
        
        # Validate specific values
        if config['sdk'] != 'gradio':
            print("❌ SDK must be 'gradio'")
            return False
        
        if config['app_file'] != 'app.py':
            print("❌ app_file must be 'app.py'")
            return False
        
        # Check for agent-demo-track tag
        tags = config.get('tags', [])
        if 'agent-demo-track' not in tags:
            print("❌ Missing 'agent-demo-track' tag")
            return False
        
        print("✅ README.md configuration is valid")
        print(f"   Title: {config['title']}")
        print(f"   Emoji: {config['emoji']}")
        print(f"   SDK: {config['sdk']}")
        print(f"   Hardware: {config.get('suggested_hardware', 'not specified')}")
        print(f"   Tags: {', '.join(tags)}")
        return True
        
    except yaml.YAMLError as e:
        print(f"❌ Invalid YAML in README.md: {e}")
        return False

def validate_app_file():
    """Validate app.py exists and has proper structure."""
    print("\n🔍 Validating app.py...")
    
    if not os.path.exists('app.py'):
        print("❌ app.py file not found")
        return False
    
    with open('app.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for Gradio import
    if 'import gradio as gr' not in content:
        print("❌ app.py missing Gradio import")
        return False
      # Check for launch call
    if '.launch(' not in content:
        print("❌ app.py missing launch() call")
        return False
    
    # Check for HF Spaces detection
    if 'SPACE_ID' in content:
        print("✅ HF Spaces detection enabled")
    else:
        print("⚠️ No HF Spaces detection found (optional)")
    
    print("✅ app.py structure is valid")
    return True

def validate_requirements():
    """Validate requirements.txt exists and has essential packages."""
    print("\n🔍 Validating requirements.txt...")
    
    if not os.path.exists('requirements.txt'):
        print("❌ requirements.txt file not found")
        return False
    
    with open('requirements.txt', 'r', encoding='utf-8') as f:
        requirements = f.read().lower()
    
    essential_packages = ['gradio', 'torch', 'transformers', 'pillow', 'openai']
    missing_packages = []
    
    for package in essential_packages:
        if package not in requirements:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing essential packages: {', '.join(missing_packages)}")
        return False
    
    print("✅ requirements.txt has all essential packages")
    return True

def validate_env_files():
    """Validate .env.example exists and .env is properly ignored."""
    print("\n🔍 Validating environment configuration...")
    
    if not os.path.exists('.env.example'):
        print("❌ .env.example file not found")
        return False
    
    # Check .gitignore exists and excludes .env
    if os.path.exists('.gitignore'):
        with open('.gitignore', 'r', encoding='utf-8') as f:
            gitignore_content = f.read()
        
        if '.env' not in gitignore_content:
            print("⚠️ .env should be in .gitignore")
        else:
            print("✅ .env properly ignored in .gitignore")
    
    print("✅ Environment configuration is valid")
    return True

def validate_imports():
    """Validate that all critical imports work correctly."""
    print("\n🔍 Validating critical imports...")
    
    # Set validation mode to prevent model loading
    import os
    os.environ['VALIDATION_MODE'] = 'true'
    
    try:
        # Test llama_agents import
        from llama_agents import get_image_analyzer, ShoppingListAgent
        print("✅ llama_agents imports successful")
        
        # Test meal_planner import
        from meal_planner import MealPlannerService
        print("✅ meal_planner imports successful")
        
        print("✅ All critical imports work correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {str(e)}")
        return False

def main():
    """Run all validation checks."""
    print("🚀 Hugging Face Spaces Deployment Validation")
    print("=" * 50)
    
    checks = [
        validate_readme(),
        validate_app_file(),
        validate_requirements(),
        validate_env_files(),
        validate_imports()
    ]
    
    if all(checks):
        print("\n🎉 All validation checks passed!")
        print("✅ Your app is ready for Hugging Face Spaces deployment!")
        print("\n📋 Next steps:")
        print("1. Go to https://huggingface.co/new-space")
        print("2. Choose 'Gradio' SDK and 'T4 Small' hardware")
        print("3. Upload your files or connect via Git")
        print("4. Add OPENAI_API_KEY in Spaces settings")
        print("5. Your app will be live at: https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME")
    else:
        print("\n❌ Some validation checks failed!")
        print("Please fix the issues above before deploying.")

if __name__ == "__main__":
    main()
