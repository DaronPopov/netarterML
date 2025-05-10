import sys
import os
import importlib

def test_import(module_name):
    try:
        module = importlib.import_module(module_name)
        print(f"✓ Successfully imported {module_name} ({module.__version__ if hasattr(module, '__version__') else 'version unknown'})")
        return True
    except ImportError as e:
        print(f"✗ Failed to import {module_name}: {e}")
        return False

def main():
    print("Python Environment Test")
    print("=" * 50)
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Current directory: {os.getcwd()}")
    print("\nPython path:")
    for path in sys.path:
        print(f"  {path}")
    
    print("\nTesting required imports:")
    modules = ['torch', 'diffusers', 'transformers', 'numpy']
    all_passed = all(test_import(module) for module in modules)
    
    print("\nEnvironment variables:")
    for key, value in os.environ.items():
        if 'PYTHON' in key:
            print(f"{key}: {value}")
    
    if all_passed:
        print("\n✓ All module imports successful!")
        sys.exit(0)
    else:
        print("\n✗ Some module imports failed")
        sys.exit(1)

if __name__ == '__main__':
    main() 