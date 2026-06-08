import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOCAL_ALPHAFOLD_SRC = os.path.join(ROOT_DIR, "alphafold", "src")

if LOCAL_ALPHAFOLD_SRC not in sys.path:
    sys.path.insert(0, LOCAL_ALPHAFOLD_SRC)
