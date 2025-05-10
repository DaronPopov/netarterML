import numpy as np
import time
import logging
import csv
import os
import sys
import pandas as pd
import asyncio
import torch
from finlib.core.asm.kernels.transformer import Transformer
from finlib.core.examples.stock_trend_analyzer import StockTrendAnalyzer

# Suppress all logs from transformer module
logging.getLogger("finlib.asm.transformer").setLevel(logging.CRITICAL)
logging.getLogger("finlib.asm.fused_transformer_op").setLevel(logging.CRITICAL)
logging.getLogger("finlib.core.examples.stock_trend_analyzer").setLevel(logging.CRITICAL)

# Configure logging for the script
logging.basicConfig(level=logging.CRITICAL)
logger = logging.getLogger(__name__)

# Create a simple output function that bypasses logging
def print_status(message):
    sys.stdout.write(message + "\n")
    sys.stdout.flush()

class DummyDataGenerator:
    """Generate dummy stock data for testing"""
    def __init__(self, context_length):
        self.context_length = context_length
    
    def generate_data(self, symbol):
        # Generate random dates for the past N days
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.Timedelta(days=self.context_length)
        dates = pd.date_range(start=start_date, end=end_date, periods=self.context_length)
        
        # Generate random stock data
        data = {
            'Date': dates,
            'Open': np.random.uniform(100, 200, size=self.context_length),
            'High': np.random.uniform(100, 200, size=self.context_length),
            'Low': np.random.uniform(100, 200, size=self.context_length),
            'Close': np.random.uniform(100, 200, size=self.context_length),
            'Volume': np.random.uniform(1000000, 5000000, size=self.context_length),
        }
        
        # Ensure High is always higher than Open/Close/Low
        for i in range(self.context_length):
            data['High'][i] = max(data['High'][i], data['Open'][i], data['Close'][i], data['Low'][i]) + 1
            data['Low'][i] = min(data['Low'][i], data['Open'][i], data['Close'][i])
        
        df = pd.DataFrame(data)
        df.set_index('Date', inplace=True)
        return df

def verify_times2_model(analyzer):
    """Verify the analyzer is using the Times2.0 model"""
    try:
        if hasattr(analyzer, 'timesfm_model'):
            model_name = analyzer.timesfm_model.__class__.__name__
            config = analyzer.timesfm_config
            print_status(f"\nMODEL VERIFICATION:")
            print_status(f"  Model type: {model_name}")
            print_status(f"  Config: {config.hidden_size}d × {config.num_attention_heads}h × {config.num_hidden_layers}l")
            print_status(f"  Is Times2.0: {'Yes, using TimesFM architecture' if model_name == 'TimesFMModel' else 'No'}")
            print_status(f"  Using ASM transformer: {'Yes' if hasattr(analyzer, 'asm_transformer') else 'No'}")
            print_status(f"  Device: {analyzer.device}")
            return True
        return False
    except Exception as e:
        print_status(f"Error verifying model: {str(e)}")
        return False

async def test_stock_analyzer_performance():
    """
    Test the stock trend analyzer performance using the transformer model.
    """
    # Large model parameters (similar to test_fused_transformer.py)
    d_model = 5000
    n_heads = 40
    n_layers = 6
    context_length = 1024
    prediction_length = 100
    batch_size = 4
    
    # Initialize the StockTrendAnalyzer with one test symbol
    symbol = "AAPL"
    symbols = [symbol]
    
    print_status("Starting test with StockTrendAnalyzer large model...")
    print_status(f"Model configuration: {d_model}d × {n_heads}h × {n_layers}l")
    print_status(f"Context length: {context_length}, Prediction length: {prediction_length}")
    
    # Suppress stdout during model initialization
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    
    analyzer = None
    try:
        # Initialize the StockTrendAnalyzer
        analyzer = StockTrendAnalyzer(
            symbols=symbols,
            context_length=context_length,
            prediction_length=prediction_length,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            update_interval=0.1,
            data_buffer_size=1000
        )
        
        # Generate mock data
        data_generator = DummyDataGenerator(context_length)
        mock_data = data_generator.generate_data(symbol)
        
        # Inject the mock data into the analyzer's buffer
        analyzer.data_buffer[symbol] = mock_data
        
        # Perform a warmup run
        _ = await analyzer.get_trend_analysis(symbol)
        
    finally:
        # Restore stdout
        sys.stdout.close()
        sys.stdout = original_stdout
    
    if not analyzer:
        print_status("Failed to initialize the StockTrendAnalyzer")
        return
    
    # Verify we're using Times2.0 model
    verify_times2_model(analyzer)
    
    print_status("\nModel initialized. Running benchmark...")
    
    # Benchmark parameters
    iterations = 1  # Single iteration for large model
    
    # Progress indicators
    start_time = time.time()
    results = []
    
    for i in range(iterations):
        # Suppress all output during execution
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        
        try:
            result = await analyzer.get_trend_analysis(symbol)
            results.append(result)
        finally:
            sys.stdout.close()
            sys.stdout = original_stdout
            
        print_status(f"Iteration {i+1}/{iterations} completed")
    
    end_time = time.time()
    
    # Calculate execution time
    total_execution_time = end_time - start_time
    avg_execution_time = total_execution_time / iterations
    
    # Calculate FLOPS (approximately)
    flops = calculate_flops(context_length, d_model, n_heads, n_layers)
    gflops = flops / avg_execution_time / 1e9
    
    # Print final results
    print_status("\n" + "="*60)
    print_status("STOCK ANALYZER PERFORMANCE RESULTS:")
    print_status(f"  Model size: {d_model}d × {n_heads}h × {n_layers}l")
    print_status(f"  Context length: {context_length}")
    print_status(f"  Total execution time: {total_execution_time:.6f} seconds")
    print_status(f"  Average execution time: {avg_execution_time:.6f} seconds")
    print_status(f"  Performance: {gflops:.2f} GFLOPS")
    
    # Print sample results
    if results:
        sample = results[0]
        print_status("\nSAMPLE PREDICTION RESULT:")
        print_status(f"  Symbol: {symbol}")
        print_status(f"  Trend direction: {sample.get('trend_direction', 'Unknown')}")
        print_status(f"  Confidence: {sample.get('confidence', 0):.2f}")
        print_status(f"  Prediction interval: {sample.get('prediction_interval', [])}")
        
        # Print more detailed prediction metrics
        print_status("\nDETAILED PREDICTION METRICS:")
        for key, value in sample.items():
            if key not in ['trend_direction', 'confidence', 'prediction_interval']:
                if isinstance(value, float):
                    print_status(f"  {key}: {value:.4f}")
                else:
                    print_status(f"  {key}: {value}")
        
        # Check for prediction scores
        if 'prediction_scores' in sample:
            print_status("\nPREDICTION SCORES:")
            scores = sample['prediction_scores']
            if isinstance(scores, dict):
                for direction, score in scores.items():
                    print_status(f"  {direction}: {score:.4f}")
        
        # Check which model was used for inference
        if 'model_used' in sample:
            print_status(f"\nModel used for inference: {sample['model_used']}")
        elif hasattr(analyzer, '_get_model_info'):
            model_info = analyzer._get_model_info()
            print_status(f"\nModel info: {model_info}")
    
    print_status("="*60)
    
    # Save results to CSV
    save_results_to_csv("stock_analyzer_performance.csv", {
        "model_type": "stock_analyzer",
        "d_model": d_model,
        "n_heads": n_heads, 
        "n_layers": n_layers,
        "context_length": context_length,
        "execution_time": avg_execution_time,
        "gflops": gflops
    })
    
    return avg_execution_time, gflops

def calculate_flops(seq_len, d_model, n_heads, n_layers):
    """
    Calculate the number of floating-point operations for a transformer model.
    """
    head_dim = d_model // n_heads
    
    # Per-layer flops
    # Self-attention: 4 * seq_len * d_model^2 (QKV projections + output projection)
    qkv_flops = 3 * seq_len * d_model * d_model
    attn_matmul_flops = n_heads * seq_len * seq_len * head_dim
    attn_output_flops = seq_len * d_model * d_model
    
    # FFN: 2 * seq_len * d_model * (4*d_model)
    ffn_flops = 2 * seq_len * d_model * (4 * d_model)
    
    # Layer norm: 5 * seq_len * d_model each (2 per layer)
    ln_flops = 2 * 5 * seq_len * d_model
    
    # Total flops per layer
    flops_per_layer = qkv_flops + attn_matmul_flops + attn_output_flops + ffn_flops + ln_flops
    
    # Total flops for all layers
    total_flops = flops_per_layer * n_layers
    
    # Additional flops for feature projection (input has 5 features)
    feature_projection_flops = seq_len * 5 * d_model
    
    return total_flops + feature_projection_flops

def save_results_to_csv(filename, results):
    """
    Save benchmark results to a CSV file.
    """
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(filename)
    
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = list(results.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(results)

async def trace_model_execution(analyzer, symbol):
    """Trace model execution and print internal steps"""
    print_status("\nTRACING MODEL EXECUTION:")
    
    # Enable debug mode if available
    if hasattr(analyzer, 'debug_mode'):
        analyzer.debug_mode = True
    
    # Patch the get_trend_analysis method to print internal steps
    original_method = analyzer.get_trend_analysis
    
    async def traced_method(sym):
        print_status(f"  Calling get_trend_analysis for {sym}")
        
        # Call PyTorch model for inference
        if hasattr(analyzer, 'timesfm_model'):
            print_status(f"  Using TimesFM model for inference")
            
            # Capture which forward method is being called
            original_forward = analyzer.timesfm_model.forward
            def traced_forward(*args, **kwargs):
                print_status(f"  TimesFM model forward pass called")
                return original_forward(*args, **kwargs)
            
            analyzer.timesfm_model.forward = traced_forward
        
        # Call ASM transformer for inference
        if hasattr(analyzer, 'asm_transformer'):
            print_status(f"  Using ASM transformer for inference")
            
            # Capture which forward method is being called
            if hasattr(analyzer.asm_transformer, 'fully_fused_forward'):
                original_ff_forward = analyzer.asm_transformer.fully_fused_forward
                def traced_ff_forward(*args, **kwargs):
                    print_status(f"  ASM transformer fully_fused_forward called")
                    return original_ff_forward(*args, **kwargs)
                analyzer.asm_transformer.fully_fused_forward = traced_ff_forward
        
        # Call the original method
        result = await original_method(sym)
        
        # Restore original methods
        if hasattr(analyzer, 'timesfm_model'):
            analyzer.timesfm_model.forward = original_forward
        
        if hasattr(analyzer, 'asm_transformer') and hasattr(analyzer.asm_transformer, 'fully_fused_forward'):
            analyzer.asm_transformer.fully_fused_forward = original_ff_forward
        
        print_status(f"  Completed get_trend_analysis")
        
        # Add model info to result
        result['model_used'] = 'TimesFM Model' if hasattr(analyzer, 'timesfm_model') else 'Unknown'
        
        return result
    
    # Replace the method
    analyzer.get_trend_analysis = traced_method
    
    # Call the method
    result = await analyzer.get_trend_analysis(symbol)
    
    # Restore the original method
    analyzer.get_trend_analysis = original_method
    
    # Disable debug mode
    if hasattr(analyzer, 'debug_mode'):
        analyzer.debug_mode = False
    
    return result

if __name__ == "__main__":
    async def main():
        execution_time, gflops = await test_stock_analyzer_performance()
        
        # Create a new analyzer for tracing the execution
        print_status("\nRunning model execution trace to verify Times2.0 model...")
        
        symbol = "AAPL"
        analyzer = StockTrendAnalyzer(
            symbols=[symbol],
            context_length=200,
            prediction_length=40,
            d_model=5000,
            n_heads=40,
            n_layers=6,
            update_interval=0.1,
            data_buffer_size=1000,
        )
        
        # Generate mock data
        data_generator = DummyDataGenerator(200)
        mock_data = data_generator.generate_data(symbol)
        
        # Inject the mock data
        analyzer.data_buffer[symbol] = mock_data
        
        # Trace the execution
        await trace_model_execution(analyzer, symbol)
        
        print_status("\nResults saved to stock_analyzer_performance.csv")
    
    asyncio.run(main()) 