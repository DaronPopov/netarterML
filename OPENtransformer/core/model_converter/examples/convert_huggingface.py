import logging
from pathlib import Path
from finlib.core.model_converter.huggingface_converter import HuggingFaceConverter
from finlib.core.asm.kernels.fused_transformer_op import create_fully_fused_transformer_op

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def convert_huggingface_model(
    model_name: str,
    output_dir: str,
    model_type: str = None,
    model_config: dict = None
):
    """
    Convert a Hugging Face model to the custom ASM transformer format.
    
    Args:
        model_name: Name of the model on Hugging Face Hub or local path
        output_dir: Directory to save the converted model
        model_type: Optional model type to specify
        model_config: Optional model configuration
    """
    try:
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize converter
        converter = HuggingFaceConverter(model_config)
        
        # Convert model
        converter.convert_model(
            model_name_or_path=model_name,
            output_dir=str(output_path),
            model_type=model_type,
            model_config=model_config
        )
        
        # Test loading the converted model
        logger.info("Testing converted model loading...")
        builder_func = create_fully_fused_transformer_op
        loader = converter.load_converted_model(str(output_path), builder_func)
        
        # Get model dimensions
        d_model, n_heads, n_layers = loader.get_model_dimensions()
        logger.info(f"Successfully loaded converted model with dimensions:")
        logger.info(f"- d_model: {d_model}")
        logger.info(f"- n_heads: {n_heads}")
        logger.info(f"- n_layers: {n_layers}")
        
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Error in conversion process: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    model_name = "bert-base-uncased"  # Example model
    output_dir = "converted_models/bert_base"
    
    # Convert model
    converted_path = convert_huggingface_model(
        model_name=model_name,
        output_dir=output_dir,
        model_type="bert",
        model_config={
            "max_seq_length": 512,
            "vocab_size": 30522
        }
    )
    
    logger.info(f"Model successfully converted and saved to {converted_path}") 