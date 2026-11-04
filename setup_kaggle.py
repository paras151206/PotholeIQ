"""
setup_kaggle.py
----------------
Helper script to set up Kaggle API credentials.

Usage:
    python setup_kaggle.py

You will be prompted for your Kaggle username and API key.
Get your API key from: https://www.kaggle.com/settings → API → Create New Token
"""

import json
import os
import sys
from pathlib import Path

def main():
    print("\n" + "="*55)
    print("  Kaggle API Credentials Setup")
    print("="*55)
    print("\nGet your API token from:")
    print("  https://www.kaggle.com/settings → API → Create New Token")
    print("  This downloads a kaggle.json file with your credentials.\n")

    kaggle_dir  = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"

    if kaggle_json.exists():
        print(f"[setup] kaggle.json already exists at {kaggle_json}")
        overwrite = input("Overwrite? (y/N): ").strip().lower()
        if overwrite != "y":
            print("[setup] Keeping existing credentials.")
            return

    username = input("Enter your Kaggle username: ").strip()
    key      = input("Enter your Kaggle API key:  ").strip()

    if not username or not key:
        print("[setup] ERROR: Both username and key are required.")
        sys.exit(1)

    kaggle_dir.mkdir(mode=0o700, exist_ok=True)
    creds = {"username": username, "key": key}
    with open(kaggle_json, "w") as f:
        json.dump(creds, f)
    kaggle_json.chmod(0o600)

    print(f"\n[setup] ✓ Credentials saved to {kaggle_json}")
    print("[setup] Now run: python src/data/download.py\n")


if __name__ == "__main__":
    main()
