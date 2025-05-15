# check_permissions.py
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data" / "jungbub_teachings"

print(f"Checking permissions for {DATA_DIR}")
if DATA_DIR.exists():
    print(f"Directory exists: {DATA_DIR}")
    print(f"Is readable: {os.access(DATA_DIR, os.R_OK)}")
    print(f"Is writable: {os.access(DATA_DIR, os.W_OK)}")
    print(f"Is executable: {os.access(DATA_DIR, os.X_OK)}")
else:
    print(f"Directory does not exist: {DATA_DIR}")