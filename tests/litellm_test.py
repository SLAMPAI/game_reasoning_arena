import os
import litellm

# Define your API keys (set them manually or via environment variables)
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "your_together_api_key")
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", "your_fireworks_api_key")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your_groq_api_key")

# Define models to test: (name used by litellm, associated key)
# Note: These are the underlying model names without the litellm_ prefix
# since this test directly calls the litellm library
MODELS = [
    # Together.ai
    ("together_ai/meta-llama/Llama-3-8b-chat-hf", TOGETHER_API_KEY),
    ("together_ai/mistralai/Mistral-7B-Instruct-v0.2", TOGETHER_API_KEY),

    # Groq
    ("groq/llama3-8b-8192", GROQ_API_KEY),
    ("groq/gemma2-9b-it", GROQ_API_KEY),

    # Fireworks
    ("fireworks_ai/accounts/fireworks/models/llama-v3-8b-instruct", FIREWORKS_API_KEY),
    ("fireworks_ai/accounts/fireworks/models/llama-v3-70b-instruct", FIREWORKS_API_KEY),
]

prompt = "What is the capital of France?"

def test_model(model_name: str, api_key: str):
    print(f"\n=== Testing {model_name} ===")
    try:
        os.environ["LITELLM_API_KEY"] = api_key
        response = litellm.completion(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
        )
        print("Response:", response["choices"][0]["message"]["content"].strip())
    except Exception as e:
        print("Error:", str(e))

if __name__ == "__main__":
    for model, key in MODELS:
        if key == "your_together_api_key" or key == "your_fireworks_api_key" or key == "your_groq_api_key":
            print(f"\n⚠️  Skipping {model} (missing API key)")
            continue
        test_model(model, key)
