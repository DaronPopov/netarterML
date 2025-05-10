import unittest
import numpy as np
import os
from OPENtransformer.arm64_engine.core.asm.kernels.vision.diffusion.text_to_video_pipeline import TextToVideoPipeline

class TestTextToVideoPipeline(unittest.TestCase):
    """Test cases for TextToVideoPipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.pipeline = TextToVideoPipeline(
            model_name="openai/clip-vit-base-patch32",
            device="cpu",
            num_inference_steps=10  # Reduced for testing
        )
        
    def test_initialization(self):
        """Test pipeline initialization."""
        self.assertIsNotNone(self.pipeline.text_encoder)
        self.assertIsNotNone(self.pipeline.video_unet)
        self.assertIsNotNone(self.pipeline.diffusion)
        
    def test_text_encoding(self):
        """Test text encoding."""
        prompt = "A cat playing with a ball"
        text_emb = self.pipeline.text_encoder.encode(prompt)
        
        # Check shape
        self.assertEqual(text_emb.shape[0], 1)  # batch size
        self.assertEqual(text_emb.shape[1], 77)  # sequence length
        self.assertEqual(text_emb.shape[2], 768)  # embedding dimension
        
    def test_video_generation(self):
        """Test video generation."""
        prompt = "A cat playing with a ball"
        video = self.pipeline.generate(
            prompt=prompt,
            num_frames=8,  # Reduced for testing
            video_size=(32, 32),  # Reduced for testing
            num_inference_steps=10  # Reduced for testing
        )
        
        # Check shape
        self.assertEqual(video.shape[0], 1)  # batch size
        self.assertEqual(video.shape[1], 3)  # RGB channels
        self.assertEqual(video.shape[2], 8)  # num frames
        self.assertEqual(video.shape[3], 32)  # height
        self.assertEqual(video.shape[4], 32)  # width
        
        # Check value range
        self.assertTrue(np.all(video >= 0) and np.all(video <= 1))
        
    def test_video_saving(self):
        """Test video saving."""
        # Generate a small test video
        prompt = "A cat playing with a ball"
        video = self.pipeline.generate(
            prompt=prompt,
            num_frames=8,
            video_size=(32, 32),
            num_inference_steps=10
        )
        
        # Save video
        output_path = "test_output.mp4"
        self.pipeline.save_video(video, output_path)
        
        # Check if file exists
        self.assertTrue(os.path.exists(output_path))
        
        # Clean up
        os.remove(output_path)

if __name__ == '__main__':
    unittest.main() 