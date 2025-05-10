import asyncio
import websockets
import json
import datetime
from typing import Dict
import argparse
import os
import sys

async def receive_updates(uri: str = "ws://localhost:8766"):
    """Connect to the WebSocket server and receive real-time stock trend updates."""
    print(f"Connecting to WebSocket server at {uri}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected! Receiving real-time stock trend updates...")
            print("Press Ctrl+C to stop.\n")
            
            # Track last update time per symbol
            last_updates = {}
            
            while True:
                try:
                    # Receive message
                    message = await websocket.recv()
                    analysis = json.loads(message)
                    
                    # Get symbol and check if there's an error
                    symbol = analysis.get("symbol", "Unknown")
                    if "error" in analysis:
                        print(f"Error for {symbol}: {analysis['error']}")
                        continue
                    
                    # Calculate time since last update
                    now = datetime.datetime.now()
                    time_since_last = ""
                    if symbol in last_updates:
                        diff = now - last_updates[symbol]
                        time_since_last = f"({diff.total_seconds():.1f}s since last update)"
                    last_updates[symbol] = now
                    
                    # Format and print the analysis
                    print(f"\n{'-'*50}")
                    print(f"SYMBOL: {symbol} {time_since_last}")
                    print(f"{'-'*50}")
                    print(f"Current Price: {analysis['current_price']:.4f}")
                    
                    # Print predicted prices
                    predicted_prices = analysis['predicted_prices']
                    print(f"Prediction (next {len(predicted_prices)} days):")
                    # Show a simple chart with +/- indicators
                    current = analysis['current_price']
                    for i, price in enumerate(predicted_prices):
                        direction = "+" if price > current else "-"
                        strength = abs(price - current) / current
                        bars = "█" * min(int(strength * 100), 20)
                        print(f"  Day {i+1}: {price:.4f} {direction} {bars}")
                    
                    # Print trend information
                    trend_emoji = "📈" if analysis['trend_direction'] == "up" else "📉"
                    print(f"Trend Direction: {analysis['trend_direction']} {trend_emoji}")
                    print(f"Trend Strength: {analysis['trend_strength']*100:.2f}%")
                    print(f"Confidence: {analysis['confidence']*100:.2f}%")
                    print(f"Last Updated: {analysis['last_updated']}")
                    
                except Exception as e:
                    print(f"Error processing message: {e}")
                    
    except KeyboardInterrupt:
        print("\nDisconnecting from server...")
    except Exception as e:
        print(f"Connection error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Stock Trend Analyzer Client")
    parser.add_argument("--host", default="localhost", help="WebSocket server host")
    parser.add_argument("--port", type=int, default=8766, help="WebSocket server port")
    args = parser.parse_args()
    
    uri = f"ws://{args.host}:{args.port}"
    
    try:
        asyncio.run(receive_updates(uri))
    except KeyboardInterrupt:
        print("\nClient stopped by user")
    
if __name__ == "__main__":
    main() 