"""
One-time token generator for GitHub Actions.
Run this locally to generate OAuth tokens and upload them as a GitHub Secret.

Usage:
    python generate_tokens.py
"""

import os
import subprocess
from dotenv import load_dotenv
from garminconnect import Garmin

load_dotenv()

email = os.getenv('GARMIN_EMAIL') or input("Garmin email: ")
password = os.getenv('GARMIN_PASSWORD') or input("Garmin password: ")

print("Logging in to Garmin Connect...")
try:
    api = Garmin(email, password)
    api.login()
    token_str = api.garth.dumps()
    print(f"✅ Login OK - got tokens ({len(token_str)} chars)")
except Exception as e:
    print(f"❌ Login failed: {e}")
    exit(1)

# Upload to GitHub Secrets via gh CLI
gh = r"C:\Program Files\GitHub CLI\gh.exe"
repo = "Dexter696/garmin-dashboard"

print(f"\nUploading GARMIN_TOKENS secret to {repo}...")
try:
    result = subprocess.run(
        [gh, "secret", "set", "GARMIN_TOKENS", "--repo", repo, "--body", token_str],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print("✅ Secret GARMIN_TOKENS set successfully!")
        print("\nYou can now push the code changes and the workflow will use tokens instead of password login.")
    else:
        print(f"❌ gh CLI error: {result.stderr}")
        print("\nManual fallback - copy this token string and set it as GitHub Secret 'GARMIN_TOKENS':")
        print("-" * 60)
        print(token_str)
        print("-" * 60)
except FileNotFoundError:
    print("❌ gh CLI not found")
    print("\nManual fallback - copy this token string and set it as GitHub Secret 'GARMIN_TOKENS':")
    print("-" * 60)
    print(token_str)
    print("-" * 60)
