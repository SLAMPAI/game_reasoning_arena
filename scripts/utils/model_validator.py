#!/usr/bin/env python3
"""
Pre-run Validation Utility

Validates models and checks API credits before running experiments.
"""

import sys
import requests
import argparse
import logging
import os
from pathlib import Path
from typing import List, Dict, Any

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env file from the project root
    env_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(env_path)
except ImportError:
    print("⚠️  python-dotenv not available, using system environment only")

# Add src to path
current_dir = Path(__file__).parent.parent.parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

from game_reasoning_arena.backends.llm_registry import (
    detect_model_backend
)
from game_reasoning_arena.backends.backend_config import (
    config as backend_config
)
from game_reasoning_arena.configs.config_parser import (
    build_cli_parser, parse_config
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_openrouter_credits(api_key: str) -> Dict[str, Any]:
    """Check OpenRouter credits and limits."""
    credits_info = {
        "accessible": False,
        "balance": None,
        "error": None,
        "warnings": []
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/credits",
            headers=headers,
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            credits_info["accessible"] = True

            # Handle the actual OpenRouter response format
            if "data" in data:
                credit_data = data["data"]
                total_credits = credit_data.get("total_credits", 0)
                total_usage = credit_data.get("total_usage", 0)
                balance = total_credits - total_usage
                credits_info["balance"] = balance

                if balance < 1.0:
                    credits_info["warnings"].append(
                        f"Low balance: ${balance:.4f}"
                    )
                elif balance < 5.0:
                    credits_info["warnings"].append(
                        f"Balance warning: ${balance:.2f}"
                    )
            else:
                # Fallback for other response formats
                credits = data.get("credits", {})
                credits_info["balance"] = credits.get("balance")
        else:
            credits_info["error"] = f"API failed: {response.status_code}"

    except requests.RequestException as e:
        credits_info["error"] = f"Request failed: {str(e)}"
    except Exception as e:
        credits_info["error"] = f"Error: {str(e)}"

    return credits_info


def validate_model(model_name: str) -> Dict[str, Any]:
    """Validate a single model."""
    result = {
        "model": model_name,
        "valid": False,
        "backend": None,
        "error": None,
        "credits_info": None
    }

    try:
        backend = detect_model_backend(model_name)
        result["backend"] = backend

        if backend == "openrouter":
            # Only validate OpenRouter models thoroughly (check credits)
            try:
                api_key = backend_config.get_api_key("openrouter")
                credits_info = check_openrouter_credits(api_key)
                result["credits_info"] = credits_info

                if credits_info["accessible"]:
                    result["valid"] = True
                else:
                    result["error"] = "Cannot access OpenRouter API"

            except ValueError as e:
                result["error"] = f"API key error: {str(e)}"

        else:
            # For all other backends (LiteLLM, vLLM, HuggingFace),
            # just assume they're valid - we only care about OpenRouter credits
            result["valid"] = True

    except Exception as e:
        result["error"] = f"Validation error: {str(e)}"

    return result


def extract_models_from_config(config: Dict[str, Any]) -> List[str]:
    """Extract all models from a configuration."""
    models = []

    # Check top-level models list
    if "models" in config and isinstance(config["models"], list):
        models.extend(config["models"])

    # Check agent configurations
    agents = config.get("agents", {})
    for agent in agents.values():
        if agent.get("type") == "llm" and "model" in agent:
            if agent["model"] not in models:
                models.append(agent["model"])

    # Check llm_backend default_model
    if "llm_backend" in config and "default_model" in config["llm_backend"]:
        default_model = config["llm_backend"]["default_model"]
        if default_model not in models:
            models.append(default_model)

    return models


def validate_models_and_credits(
    models: List[str],
    show_details: bool = True
) -> Dict[str, Any]:
    """Validate a list of models and check credits."""
    results = {
        "all_valid": True,
        "models": {},
        "summary": {
            "total": len(models),
            "valid": 0,
            "invalid": 0,
            "credits_warnings": []
        }
    }

    if show_details:
        print(f"\n🔍 Validating {len(models)} models...")
        print("   (Only checking OpenRouter credits - "
              "other models assumed valid)")
        print("=" * 60)

    for model in models:
        validation_result = validate_model(model)
        results["models"][model] = validation_result

        if validation_result["valid"]:
            results["summary"]["valid"] += 1
            if show_details:
                backend = validation_result.get("backend", "unknown")
                print(f"✅ {model:<40} ({backend})")

                # Show credits info for OpenRouter models
                credits = validation_result.get("credits_info")
                if credits:
                    if credits["balance"] is not None:
                        print(f"   💰 Balance: ${credits['balance']:.4f}")
                    if credits["warnings"]:
                        for warning in credits["warnings"]:
                            print(f"   ⚠️  {warning}")
                            warning_msg = f"{model}: {warning}"
                            results["summary"]["credits_warnings"].append(
                                warning_msg
                            )
                    # Debug: show if credits check failed
                    if not credits["accessible"]:
                        error = credits.get('error', 'Unknown')
                        print(f"   ⚠️  Credits check failed: {error}")
        else:
            results["all_valid"] = False
            results["summary"]["invalid"] += 1
            if show_details:
                error = validation_result.get("error", "Unknown error")
                print(f"❌ {model:<40} - {error}")

    if show_details:
        print("=" * 60)
        valid = results['summary']['valid']
        total = results['summary']['total']
        print(f"📊 Summary: {valid}/{total} models valid")

        if results["summary"]["credits_warnings"]:
            print("\n⚠️  Credit Warnings:")
            for warning in results["summary"]["credits_warnings"]:
                print(f"   {warning}")

        if not results["all_valid"]:
            invalid = results['summary']['invalid']
            print(f"\n❌ {invalid} invalid models detected!")
            print("   Fix these issues before running experiments.")
        else:
            print("\n✅ All models validated successfully!")

    return results


def main():
    """Main function for standalone usage."""
    parser = argparse.ArgumentParser(
        description="Validate models and check credits"
    )
    parser.add_argument("--config", help="Configuration file to validate")
    parser.add_argument("--models", nargs="+", help="Specific models")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress output")

    args = parser.parse_args()

    models = []

    if args.models:
        models = args.models
    elif args.config:
        config_parser = build_cli_parser()
        config_args = config_parser.parse_args(["--config", args.config])
        config = parse_config(config_args)
        models = extract_models_from_config(config)
    else:
        print("Error: Must provide either --config or --models")
        return 1

    if not models:
        print("Error: No models found to validate")
        return 1

    validation_result = validate_models_and_credits(
        models, show_details=not args.quiet
    )

    if validation_result["all_valid"]:
        if not args.quiet:
            print("\n🎉 Ready to run experiments!")
        return 0
    else:
        if not args.quiet:
            print("\n❌ Fix validation issues before running")
        return 1


if __name__ == "__main__":
    sys.exit(main())
