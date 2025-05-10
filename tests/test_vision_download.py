#!/usr/bin/env python3

import sys
from pathlib import Path
import unittest

# Add project root to Python path (now two levels up from test file)
project_root = str(Path(__file__).absolute().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class TestImports(unittest.TestCase):
    def test_diffusion_imports(self):
        """Test diffusion model imports and initialization"""
        try:
            from OPENtransformer.diffusion.easy_diffusion_api import EasyDiffusionAPI
            api = EasyDiffusionAPI()
            self.assertIsNotNone(api)
            print("✓ Successfully imported and initialized EasyDiffusionAPI")
        except Exception as e:
            self.fail(f"Failed to import EasyDiffusionAPI: {e}")

    def test_vision_imports(self):
        """Test vision model imports and initialization"""
        try:
            from OPENtransformer.vision.easy_image import EasyImage
            api = EasyImage()
            self.assertIsNotNone(api)
            print("✓ Successfully imported and initialized EasyImage")
        except Exception as e:
            self.fail(f"Failed to import EasyImage: {e}")

    def test_llm_imports(self):
        """Test LLM model imports and initialization"""
        try:
            from OPENtransformer.chat.llm_api import LLMAPI
            # Initialize with a default model name
            api = LLMAPI(model_name="gpt2")
            self.assertIsNotNone(api)
            print("✓ Successfully imported and initialized LLMAPI")
        except Exception as e:
            self.fail(f"Failed to import LLMAPI: {e}")

    def test_core_imports(self):
        """Test core transformer imports"""
        try:
            # Import specific kernels instead of the generic transformer module
            from OPENtransformer.arm64_engine.core.asm.kernels.diffusion.fp32_optimized.cross_attention_kernel_asm import CrossAttentionKernelASM
            from OPENtransformer.arm64_engine.core.asm.kernels.diffusion.fp32_optimized.memory_efficient_attention_kernel_asm import MemoryEfficientAttentionKernelASM
            self.assertIsNotNone(CrossAttentionKernelASM)
            self.assertIsNotNone(MemoryEfficientAttentionKernelASM)
            print("✓ Successfully imported core transformer kernels")
        except Exception as e:
            self.fail(f"Failed to import core transformer kernels: {e}")

def main():
    # Run all tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

if __name__ == '__main__':
    main() 