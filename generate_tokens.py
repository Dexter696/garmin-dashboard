"""
One-time token generator for GitHub Actions.
Run this locally to generate OAuth tokens (handles MFA) and:
  1. Saves them to .env as GARMIN_TOKENS (for local use)
  2. Uploads them as GitHub Secret GARMIN_TOKENS (for GitHub Actions)

Usage:
    python generate_tokens.py
"""

import os
import sys
import subprocess
from dotenv import load_dotenv
from garminconnect import Garmin

# Force UTF-8 so emojis work on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

email = os.getenv('GARMIN_EMAIL') or input("Garmin email: ")
password = os.getenv('GARMIN_PASSWORD') or input("Garmin password: ")

print("Logging in to Garmin Connect...")
print("(If MFA is required, check your email/phone for a code)")
try:
    api = Garmin(email, password, prompt_mfa=lambda: input("Enter MFA code: "))
    api.login()
    token_str = api.client.dumps()
    print(f"Login OK - got tokens ({len(token_str)} chars)")
except Exception as e:
    print(f"Login failed: {type(e).__name__}: {e}")
    sys.exit(1)

# --- Save to .env ---
env_path = os.path.join(os.path.dirname(__file__), '.env')
try:
    with open(env_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Replace existing GARMIN_TOKENS line or append
    token_line = f'GARMIN_TOKENS={token_str}\n'
    found = False
    for i, line in enumerate(lines):
        if line.startswith('GARMIN_TOKENS='):
            lines[i] = token_line
            found = True
            break
    if not found:
        lines.append(token_line)

    with open(env_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print("Tokens saved to .env")
except Exception as e:
    print(f"Could not save to .env: {e}")

# --- Upload to GitHub Secrets ---
gh = r"C:\Program Files\GitHub CLI\gh.exe"
repo = "Dexter696/garmin-dashboard"

print(f"\nUploading GARMIN_TOKENS secret to {repo}...")
try:
    result = subprocess.run(
        [gh, "secret", "set", "GARMIN_TOKENS", "--repo", repo, "--body", token_str],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print("Secret GARMIN_TOKENS set on GitHub!")
        print("\nEverything is ready. The daily sync will use saved tokens.")
    else:
        print(f"gh CLI error: {result.stderr}")
        print("\nManual fallback - set this as GitHub Secret 'GARMIN_TOKENS':")
        print("-" * 60)
        print(token_str)
        print("-" * 60)
except FileNotFoundError:
    print("gh CLI not found")
    print("\nManual fallback - set this as GitHub Secret 'GARMIN_TOKENS':")
    print("-" * 60)
    print(token_str)
    print("-" * 60)
