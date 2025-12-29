#!/usr/bin/env python3
"""Setup script for Virtual Health Assistant."""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e.stderr}")
        return False

def main():
    """Main setup function."""
    print("🏥 Virtual Health Assistant - Setup Script")
    print("=" * 50)
    print()
    print("⚠️  DISCLAIMER: This is a research demonstration only.")
    print("   NOT FOR CLINICAL USE - Do not use for medical decisions.")
    print()
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Create necessary directories
    directories = ['data', 'models', 'assets', 'logs']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Run basic tests
    if not run_command("python -c \"import sys; sys.path.append('src'); from models import ClinicalBERTClassifier; print('Import test successful')\"", "Testing imports"):
        print("❌ Import test failed")
        sys.exit(1)
    
    print()
    print("🎉 Setup completed successfully!")
    print()
    print("Next steps:")
    print("1. Train models: python scripts/train.py")
    print("2. Run evaluation: python scripts/evaluate.py")
    print("3. Launch demo: streamlit run demo/app.py")
    print("4. Run tests: pytest tests/")
    print()
    print("📚 Documentation: README.md")
    print("⚠️  Safety information: DISCLAIMER.md")

if __name__ == "__main__":
    main()
