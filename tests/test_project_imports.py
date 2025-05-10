#!/usr/bin/env python3

import sys
from pathlib import Path

# Add project root to Python path (now two levels up from test file)
project_root = str(Path(__file__).absolute().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_project_imports():
    try:
        from OPENtransformer import __init__
        print("✓ OPENtransformer package imported successfully")
    except ImportError as e:
        print("✗ OPENtransformer package import failed:", str(e))

    try:
        from OPENtransformer.diffusion import easy_diffusion_api
        print("✓ Diffusion API imported successfully")
    except ImportError as e:
        print("✗ Diffusion API import failed:", str(e))

    try:
        from OPENtransformer.vision import easy_image
        print("✓ Vision API imported successfully")
    except ImportError as e:
        print("✗ Vision API import failed:", str(e))

    try:
        from OPENtransformer.chat import llm_api
        print("✓ LLM API imported successfully")
    except ImportError as e:
        print("✗ LLM API import failed:", str(e))

    try:
        from OPENtransformer.core import asm
        print("✓ Core ASM module imported successfully")
    except ImportError as e:
        print("✗ Core ASM module import failed:", str(e))

    try:
        from OPENtransformer.utils import dataset
        print("✓ Utils module imported successfully")
    except ImportError as e:
        print("✗ Utils module import failed:", str(e))

if __name__ == "__main__":
    print("Testing project imports...")
    test_project_imports() 