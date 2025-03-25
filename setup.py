#!/usr/bin/env python3
"""
Setup script for the AI Detection Tool.
This will install dependencies and prepare the environment.
"""

import os
import sys
import subprocess
import ssl
import nltk
from pathlib import Path

def setup_environment():
    """Set up the environment for the AI Detection Tool."""
    print("Setting up environment for AI Detection Tool...")
    
    # Create necessary directories
    os.makedirs('llm_response_cache', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Install dependencies
    print("\nInstalling dependencies...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
    
    # Fix NLTK SSL certificate issue and download required data
    print("\nDownloading NLTK data...")
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    nltk.download('punkt')
    nltk.download('stopwords')
    
    # Create sample .env file if it doesn't exist
    if not os.path.exists('.env'):
        print("\nCreating sample .env file...")
        with open('.env', 'w') as f:
            f.write("""# API Keys for LLM providers
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GOOGLE_API_KEY=your_google_key_here
""")
        print("Please edit the .env file with your actual API keys.")
    
    # Create example questions file
    if not os.path.exists('questions.txt'):
        print("\nCreating sample questions file...")
        with open('questions.txt', 'w') as f:
            f.write("""Explain the concept of natural selection and how it leads to evolution.
What are the major causes and effects of climate change?
Analyze the themes in Shakespeare's Hamlet.
Describe the key principles of macroeconomics.
""")
    
    print("\nSetup completed successfully!")
    print("\nNext steps:")
    print("1. Add your API keys to the .env file")
    print("2. Place student submissions (Word documents) in a directory")
    print("3. Run the tool with: python ai_detection.py --questions questions.txt --submissions your_submissions_dir")

if __name__ == "__main__":
    setup_environment()