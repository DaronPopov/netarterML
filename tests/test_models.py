import pytest
import os
from pathlib import Path

def test_model_download():
    """Test model downloading functionality."""
    # Run the download command
    import subprocess
    result = subprocess.run(
        ["./run_demo.sh", "download"],
        capture_output=True,
        text=True
    )
    
    # Check if the command executed successfully
    assert result.returncode == 0, f"Command failed with error: {result.stderr}"
    
    # Check if the models directory exists
    models_dir = Path("models")
    assert models_dir.exists(), "Models directory not found"
    
    # Check if required model directories exist
    required_models = [
        "vision/vit-base",
        "vision/stable-diffusion-v1-4",
        "language/tinyllama-1.1b-chat",
        "medical/vit-xray-pneumonia"
    ]
    
    for model_path in required_models:
        assert (models_dir / model_path).exists(), f"Model {model_path} not found"

def test_model_list():
    """Test model listing functionality."""
    # Run the list command
    import subprocess
    result = subprocess.run(
        ["./run_demo.sh", "list"],
        capture_output=True,
        text=True
    )
    
    # Check if the command executed successfully
    assert result.returncode == 0, f"Command failed with error: {result.stderr}"
    
    # Check if the output contains expected information
    assert "Available models:" in result.stdout, "No model list found in output" 