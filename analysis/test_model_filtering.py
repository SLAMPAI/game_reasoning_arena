#!/usr/bin/env python3
"""
Test Model Filtering Functionality

This script demonstrates and tests the model filtering functionality
for aggregate plots in the Game Reasoning Arena.
"""

import sys
from pathlib import Path

# Add analysis directory to path
sys.path.append(str(Path(__file__).parent))

from model_filtering import (
    filter_models_for_aggregate_plot,
    load_priority_models_config,
    should_filter_aggregate_plots,
    get_aggregate_plot_title_suffix
)

def test_model_filtering():
    """Test the model filtering functionality with example data."""
    print("🧪 Testing Model Filtering for Aggregate Plots")
    print("=" * 60)

    # Example models that might be in a dataset
    example_models = [
        "llm_openrouter_openai_gpt_4o_mini",
        "llm_litellm_groq_llama3_8b_8192",
        "llm_litellm_groq_llama3_70b_8192",
        "llm_litellm_together_ai_meta_llama_Meta_Llama_3.1_8B_Instruct_Turbo",
        "llm_litellm_together_ai_mistralai_Mixtral_8x7B_Instruct_v0.1",
        "llm_codegemma_7b_it",
        "llm_vllm_Qwen2_7B_Instruct",
        "llm_openrouter_google_gemini_2.0_flash_exp",
        "llm_anthropic_claude_3_haiku",
        "llm_openai_gpt_4o",
        "random_agent_42",
        "random_agent_123"
    ]

    print(f"📊 Available models ({len(example_models)}):")
    for i, model in enumerate(example_models, 1):
        print(f"  {i:2d}. {model}")

    print(f"\n🔍 Loading priority models configuration...")
    config = load_priority_models_config()
    max_models = config.get("max_models_in_aggregate", 7)

    print(f"📏 Configuration:")
    print(f"  • Max models in aggregate plots: {max_models}")
    print(f"  • Priority models: {len(config.get('priority_models', []))}")
    print(f"  • Exclude patterns: {config.get('exclude_patterns', [])}")

    print(f"\n🎯 Should filter? {should_filter_aggregate_plots(example_models)}")

    print(f"\n🔧 Applying model filtering...")
    filtered_models = filter_models_for_aggregate_plot(
        example_models,
        max_models=max_models,
        priority_config=config
    )

    print(f"\n✅ Filtered models for aggregate plots ({len(filtered_models)}):")
    for i, model in enumerate(filtered_models, 1):
        print(f"  {i:2d}. {model}")

    # Test title suffix
    suffix = get_aggregate_plot_title_suffix(
        len([m for m in example_models if not m.startswith("random")]),
        len(filtered_models)
    )

    if suffix:
        print(f"\n📝 Plot title suffix: '{suffix}'")
    else:
        print(f"\n📝 No title suffix needed (showing all models)")

    print(f"\n🏁 Model filtering test completed!")

def test_different_limits():
    """Test filtering with different max model limits."""
    print(f"\n🔬 Testing Different Model Limits")
    print("=" * 60)

    example_models = [
        "gpt-4o-mini", "llama-3.1-8b", "llama-3.1-70b",
        "gemma-2-9b", "gemini-2.0-flash", "qwen-2.5-72b",
        "mistral-7b", "claude-3-haiku", "phi-3-mini",
        "codegemma-7b", "yi-34b", "deepseek-coder"
    ]

    for max_models in [3, 5, 7, 10, 15]:
        filtered = filter_models_for_aggregate_plot(
            example_models,
            max_models=max_models
        )
        print(f"  Max {max_models:2d}: {len(filtered):2d} models → {filtered}")

if __name__ == "__main__":
    test_model_filtering()
    test_different_limits()
