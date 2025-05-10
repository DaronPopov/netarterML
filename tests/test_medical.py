import pytest
import os
from pathlib import Path

def test_medical_image_analysis():
    """Test medical image analysis functionality."""
    # Check if the test image exists
    test_image = Path("local_images/normal_1.jpg")
    assert test_image.exists(), "Test image not found"
    
    # Run the medical analysis command
    import subprocess
    result = subprocess.run(
        ["./run_demo.sh", "medical", "--image", str(test_image)],
        capture_output=True,
        text=True
    )
    
    # Check if the command executed successfully
    assert result.returncode == 0, f"Command failed with error: {result.stderr}"
    
    # Check if the output contains expected information
    assert "Prediction:" in result.stdout, "No prediction found in output"
    assert "Confidence:" in result.stdout, "No confidence score found in output"

def test_medical_image_not_found():
    """Test medical image analysis with non-existent image."""
    # Run the medical analysis command with non-existent image
    import subprocess
    result = subprocess.run(
        ["./run_demo.sh", "medical", "--image", "non_existent.jpg"],
        capture_output=True,
        text=True
    )
    
    # Check if the command failed as expected
    assert result.returncode != 0, "Command should fail with non-existent image"
    assert "Error" in result.stderr, "No error message found in output" 