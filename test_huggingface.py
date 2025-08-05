"""
Simple script to test HuggingFace integration with Board Game Arena.
This minimal example demonstrates how to query the HuggingFace models.
"""

import os
import sys
import requests
from pathlib import Path

# Check if HuggingFace API is available
def check_huggingface_api():
    """Check if the HuggingFace Inference API is accessible."""
    try:
        response = requests.get(
            "https://api-inference.huggingface.co/status",
            timeout=5
        )
        return response.status_code == 200
    except Exception as e:
        print(f"Error checking HuggingFace API: {e}")
        return False

# Query HuggingFace's inference API
def query_huggingface_model(model_id, prompt, max_length=100):
    """Query a model through HuggingFace's inference API."""
    try:
        # HuggingFace Inference API endpoint
        api_url = f"https://api-inference.huggingface.co/models/{model_id}"

        # Headers with API token if provided
        headers = {}
        hf_token = os.environ.get("HUGGINGFACE_TOKEN")
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"

        # Parameters vary by model type
        if "t5" in model_id.lower():
            # T5 models expect direct input without specific formatting
            payload = {"inputs": prompt}
        else:
            # For generative models like GPT-2, BLOOM
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": max_length,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "do_sample": True,
                    "return_full_text": False
                }
            }

        print(f"Sending request to {api_url}...")
        response = requests.post(api_url, headers=headers, json=payload)

        if response.status_code == 200:
            result = response.json()
            print(f"Received response: {result}")

            # Handle different response formats
            if isinstance(result, list) and len(result) > 0:
                if "generated_text" in result[0]:
                    return result[0]["generated_text"]
                else:
                    return str(result[0])
            else:
                return str(result)
        else:
            return f"Error: {response.status_code}, {response.text}"

    except Exception as e:
        return f"Failed to query model: {str(e)}"

def main():
    print("\n🎮 Board Game Arena - HuggingFace API Test")
    print("-" * 50)
    
    # Check if HuggingFace API is accessible
    print("Checking HuggingFace API availability...")
    if check_huggingface_api():
        print("✓ HuggingFace API is accessible")
        
        # Test free models
        models_to_test = ["gpt2", "bigscience/bloom-560m", "google/flan-t5-small"]
        
        for model_id in models_to_test:
            print(f"\nTesting model: {model_id}")
            
            # Create a simple test prompt for a tic-tac-toe game
            test_prompt = f"""
You are playing a game of tic-tac-toe.
Board state:
- - -
- - -
- - -
Valid moves: 0, 1, 2, 3, 4, 5, 6, 7, 8
Choose one move from the valid moves. Only respond with the move number.
"""
            
            print(f"Prompt:\n{test_prompt}")
            response = query_huggingface_model(model_id, test_prompt, max_length=10)
            print(f"Response: {response}")
            
            # Check if the response contains a valid move
            valid_moves = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]
            valid_move_found = any(move in response for move in valid_moves)
            if valid_move_found:
                print(f"✓ Valid move found in response")
            else:
                print(f"✗ No valid move found in response")
    else:
        print("✗ HuggingFace API is not accessible")
        print("Please check your internet connection.")
    
    print("\nTest completed!")

if __name__ == "__main__":
    main()
