#!/usr/bin/env python3

import sys
from pathlib import Path

# Add project root to Python path (now two levels up from test file)
project_root = str(Path(__file__).absolute().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add the OPENtransformer/arm64_engine/core/c_inference directory to Python path
inference_path = str(Path(__file__).absolute().parent.parent / "OPENtransformer" / "arm64_engine" / "core" / "c_inference")
if inference_path not in sys.path:
    sys.path.insert(0, inference_path)

# Try importing the module
try:
    from examples.diffusion.easy_diffusion_api import EasyDiffusionAPI
    print("Successfully imported EasyDiffusionAPI")
    
    # Test initialization
    api = EasyDiffusionAPI()
    print("Successfully initialized EasyDiffusionAPI")
except Exception as e:
    print(f"Error importing EasyDiffusionAPI: {e}")
    raise 