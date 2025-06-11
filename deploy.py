#!/usr/bin/env python3
"""
Deployment helper script for Hugging Face Spaces
"""

import os
import subprocess
import sys

def check_requirements():
    """Check if all requirements are met for deployment."""
    print("🔍 Checking deployment requirements...")
    
    # Check if .env.example exists
    if not os.path.exists('.env.example'):
        print("❌ .env.example file missing")
        return False
      # Check if requirements.txt exists
    if not os.path.exists('requirements.txt'):
        print("❌ requirements.txt file missing")
        return False
    
    # Check if README.md has proper YAML frontmatter
    if not os.path.exists('README.md'):
        print("❌ README.md file missing")
        return False
    
    with open('README.md', 'r', encoding='utf-8') as f:
        readme_content = f.read()
    
    if not readme_content.startswith('---'):
        print("❌ README.md missing YAML frontmatter")
        return False
    
    # Check core files
    core_files = ['app.py', 'llama_agents.py', 'meal_planner.py']
    for file in core_files:
        if not os.path.exists(file):
            print(f"❌ {file} missing")
            return False
    
    print("✅ All deployment requirements met")
    return True

def prepare_for_spaces():
    """Prepare the repository for Hugging Face Spaces deployment."""
    print("📦 Preparing for Hugging Face Spaces deployment...")
    
    # Create or update .env.example
    env_example_content = """# OpenAI API Key for meal planning feature
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Custom model cache directory
TRANSFORMERS_CACHE=./model_cache
HF_HOME=./hf_cache
"""
    
    with open('.env.example', 'w') as f:
        f.write(env_example_content)
    print("✅ .env.example updated")
    
    # Instructions for deployment
    instructions = """
🚀 Hugging Face Spaces Deployment Instructions:

1. Create a new Space on Hugging Face:
   - Go to https://huggingface.co/new-space
   - Choose 'Gradio' as SDK
   - Set hardware to 'T4 Small' or higher for GPU support

2. Upload files or connect via Git:
   - Upload all files from this directory
   - Or push to a GitHub repo and connect it

3. Configure Secrets in Space settings:
   - Add OPENAI_API_KEY with your OpenAI API key
   - This enables the meal planning feature

4. The app will automatically:
   - Detect GPU availability
   - Download and cache the Qwen2.5-VL model
   - Provide intelligent fallback if needed

5. Share your Space:
   - Your app will be available at: https://huggingface.co/spaces/USERNAME/SPACE_NAME
   - Add the tag 'agent-demo-track' to showcase the agent capabilities

🏷️ This project is part of the agent-demo-track showcase!
"""
    
    print(instructions)

def main():
    """Main deployment preparation function."""
    if not check_requirements():
        print("❌ Deployment preparation failed")
        sys.exit(1)
    
    prepare_for_spaces()
    print("✅ Repository ready for Hugging Face Spaces deployment!")

if __name__ == "__main__":
    main()
