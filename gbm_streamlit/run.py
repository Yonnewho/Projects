#!/usr/bin/env python3
"""
Quick start script to set up and run the GBM Streamlit app.
Run this file to automatically install dependencies and launch the app.
"""

import subprocess
import sys
import os


def run_command(command, description):
    """Run a shell command and report results."""
    print(f"\n{'='*60}")
    print(f"📌 {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Main setup and launch function."""
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║          🚀 GBM Stock Market Simulator Setup               ║
    ║                                                            ║
    ║  Welcome! This script will install dependencies and      ║
    ║  launch the Streamlit application.                       ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    # Step 1: Check Python version
    print("📍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required. You have {version.major}.{version.minor}")
        sys.exit(1)
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    
    # Step 2: Install requirements
    success = run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing Dependencies"
    )
    
    if not success:
        print("\n⚠️  Some dependencies may not have installed properly.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Step 3: Launch Streamlit
    print(f"\n{'='*60}")
    print("✨ Launching Streamlit app...")
    print(f"{'='*60}")
    print("\n🌐 The app will open at: http://localhost:8501")
    print("📖 To stop the server, press Ctrl+C\n")
    
    os.system(f"{sys.executable} -m streamlit run app.py")


if __name__ == '__main__':
    main()
