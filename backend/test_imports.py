#!/usr/bin/env python3
"""Test backend imports"""

print("Testing imports...")

try:
    import fastapi
    print("✓ FastAPI imported successfully")
except ImportError as e:
    print(f"✗ FastAPI import failed: {e}")

try:
    import app.main
    print("✓ Backend app imported successfully")
except ImportError as e:
    print(f"✗ Backend app import failed: {e}")

print("Import test complete!")