LLM Backends
============

Board Game Arena supports multiple LLM inference backends, allowing you to use both API-based and locally-hosted models seamlessly. This flexibility enables mixing different providers in the same experiment and choosing the most appropriate backend for your research needs.

Overview
--------

The framework provides two main backend types:

* **LiteLLM Backend**: API-based inference supporting 100+ providers
* **vLLM Backend**: Local GPU inference for self-hosted models

Both backends implement the same interface, making them interchangeable in experiments and allowing for easy comparison between different models and deployment strategies.

LiteLLM Backend
---------------

The LiteLLM backend provides access to **100+ language models** through a unified API interface, supporting major providers including OpenAI, Anthropic, Google, Groq, Together AI, and many others.

Key Features
~~~~~~~~~~~~

* **Unified API**: Single interface for multiple providers
* **Cost-effective**: Access to free and low-cost API endpoints
* **Fast inference**: Providers like Groq offer extremely fast response times
* **No local setup**: No GPU requirements or model downloads
* **Wide model selection**: From small efficient models to large frontier models

Supported Providers
~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   ✓ OpenAI (GPT-3.5, GPT-4, GPT-4 Turbo)
   ✓ Anthropic (Claude 3, Claude 3.5 Sonnet)
   ✓ Google (Gemini Pro, Gemma)
   ✓ Groq (Llama 3, Gemma - ultra-fast inference)
   ✓ Together AI (Llama 3.1, Mixtral, Code Llama)
   ✓ Fireworks AI (Llama, Mixtral models)
   ✓ Cohere (Command models)
   ✓ Hugging Face Inference API
   ✓ Replicate (Various open-source models)
   ✓ And 90+ more providers...

Configuration
~~~~~~~~~~~~~

Models are configured in ``src/board_game_arena/configs/litellm_models.yaml``:

.. code-block:: yaml

   models:
     # OpenAI models
     - litellm_gpt-3.5-turbo
     - litellm_gpt-4
     - litellm_gpt-4-turbo

     # Groq models (fast inference)
     - litellm_groq/llama3-8b-8192
     - litellm_groq/llama3-70b-8192
     - litellm_groq/gemma-7b-it

     # Together AI models
     - litellm_together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo
     - litellm_together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1

Model Naming Convention
~~~~~~~~~~~~~~~~~~~~~~~

LiteLLM models use the prefix ``litellm_`` followed by the provider and model name:

.. code-block:: text

   Format: litellm_<provider>/<model_name>

   Examples:
   - litellm_groq/llama3-8b-8192
   - litellm_gpt-4-turbo
   - litellm_together_ai/meta-llama/Llama-2-7b-chat-hf

API Keys Setup
~~~~~~~~~~~~~~

Create a ``.env`` file in the project root with your API keys:

.. code-block:: bash

   # OpenAI
   OPENAI_API_KEY=your_openai_key_here

   # Groq (free tier available)
   GROQ_API_KEY=your_groq_key_here

   # Together AI
   TOGETHER_API_KEY=your_together_key_here

   # Fireworks AI
   FIREWORKS_API_KEY=your_fireworks_key_here

   # Anthropic
   ANTHROPIC_API_KEY=your_anthropic_key_here

Usage Example
~~~~~~~~~~~~~

.. code-block:: bash

   # Use GPT-4 via OpenAI
   python scripts/runner.py --config configs/example_config.yaml --override \\
     agents.player_0.model=litellm_gpt-4

   # Use Llama 3 via Groq (fast inference)
   python scripts/runner.py --config configs/example_config.yaml --override \\
     agents.player_0.model=litellm_groq/llama3-8b-8192

vLLM Backend
------------

The vLLM backend enables **local GPU inference** for self-hosted models, providing full control over model deployment, privacy, and customization.

Key Features
~~~~~~~~~~~~

* **Local deployment**: Complete control over model hosting
* **GPU acceleration**: Optimized inference on NVIDIA GPUs
* **Privacy**: No data leaves your infrastructure
* **Customization**: Fine-tuned models and custom configurations
* **Cost control**: No per-token API costs for heavy usage
* **Offline capability**: Works without internet connectivity

Requirements
~~~~~~~~~~~~

.. code-block:: text

   ✓ NVIDIA GPU with CUDA support
   ✓ Sufficient GPU memory (varies by model size)
   ✓ Local model files (Hugging Face format)
   ✓ vLLM package installation

Model Setup
~~~~~~~~~~~

1. **Download Models**: Obtain model files locally

.. code-block:: bash

   # Example: Download Qwen2-7B-Instruct
   git lfs clone https://huggingface.co/Qwen/Qwen2-7B-Instruct /path/to/models/Qwen2-7B-Instruct

2. **Configure Model Paths**: Update ``src/board_game_arena/configs/vllm_models.yaml``

.. code-block:: yaml

   models:
     - name: vllm_Qwen2-7B-Instruct
       model_path: /absolute/path/to/models/Qwen/Qwen2-7B-Instruct
       tokenizer_path: /absolute/path/to/models/Qwen/Qwen2-7B-Instruct
       description: Qwen2 7B Instruct model for local inference

     - name: vllm_Llama-2-7b-chat-hf
       model_path: /absolute/path/to/models/meta-llama/Llama-2-7b-chat-hf
       description: Llama2 7B Chat model

.. important::
   **All model paths must be absolute paths** to the model directories containing the model files and tokenizer.

Model Naming Convention
~~~~~~~~~~~~~~~~~~~~~~~

vLLM models use the prefix ``vllm_`` followed by the model identifier:

.. code-block:: text

   Format: vllm_<model_identifier>

   Examples:
   - vllm_Qwen2-7B-Instruct
   - vllm_Llama-2-7b-chat-hf
   - vllm_CodeLlama-7b-Instruct-hf

Usage Example
~~~~~~~~~~~~~

.. code-block:: bash

   # Use local Qwen2-7B model
   python scripts/runner.py --config configs/example_config.yaml --override \\
     agents.player_0.model=vllm_Qwen2-7B-Instruct

   # Use local Llama model
   python scripts/runner.py --config configs/example_config.yaml --override \\
     agents.player_0.model=vllm_Llama-2-7b-chat-hf

Installation
~~~~~~~~~~~~

Install vLLM package for local inference:

.. code-block:: bash

   # Install vLLM
   pip install vllm

   # For specific CUDA versions, see vLLM documentation
   pip install vllm-nightly  # Latest features

Mixed Backend Usage
-------------------

One of the powerful features of Board Game Arena is the ability to **mix different backends** in the same experiment, enabling direct comparison between API-based and local models.

LiteLLM vs vLLM Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Compare API model vs local model
   python scripts/runner.py --config configs/example_config.yaml --override \\
     mode=llm_vs_llm \\
     agents.player_0.model=litellm_groq/llama3-8b-8192 \\
     agents.player_1.model=vllm_Qwen2-7B-Instruct \\
     num_episodes=10

Cross-Provider Experiments
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Mix different API providers
   python scripts/runner.py --config configs/example_config.yaml --override \\
     mode=llm_vs_llm \\
     agents.player_0.model=litellm_gpt-4-turbo \\
     agents.player_1.model=litellm_groq/llama3-70b-8192

   # Compare API efficiency vs local control
   python scripts/runner.py --config configs/example_config.yaml --override \\
     mode=llm_vs_llm \\
     agents.player_0.model=litellm_together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct \\
     agents.player_1.model=vllm_Llama-2-7b-chat-hf

Backend Selection Guide
-----------------------

Choose the appropriate backend based on your research needs:

LiteLLM When:
~~~~~~~~~~~~~

* **Quick prototyping** and experimentation
* **Limited GPU resources** or no local hardware
* **Comparing multiple models** without setup overhead
* **Cost-effective research** with free tiers (e.g., Groq)
* **Access to frontier models** (GPT-4, Claude 3.5)
* **Fast iteration** on experiments

vLLM When:
~~~~~~~~~~

* **Privacy requirements** for sensitive data
* **High-volume experiments** where API costs become prohibitive
* **Custom model fine-tuning** and specialized deployments
* **Offline environments** without internet access
* **Full control** over inference parameters and optimization
* **Research on model behavior** requiring deterministic setups

Performance Considerations
--------------------------

Inference Speed
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 30 25 25

   * - Backend
     - Typical Latency
     - Throughput
     - Best For
   * - Groq (LiteLLM)
     - 50-200ms
     - Very High
     - Fast experimentation
   * - OpenAI (LiteLLM)
     - 500-2000ms
     - High
     - Quality baseline
   * - Local vLLM
     - 100-1000ms
     - Variable
     - Privacy, control

Cost Comparison
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Model Type
     - Setup Cost
     - Per-Token Cost
     - Break-Even Point
   * - LiteLLM API
     - $0
     - $0.001-0.01
     - < 1M tokens
   * - Local vLLM
     - GPU hardware
     - Electricity only
     - > 1M tokens

Troubleshooting
---------------

Common LiteLLM Issues
~~~~~~~~~~~~~~~~~~~~~

**Authentication Errors**:

.. code-block:: bash

   # Check API key is set
   echo $OPENAI_API_KEY

   # Verify .env file exists and is formatted correctly
   cat .env

**Rate Limiting**:

.. code-block:: bash

   # Use multiple providers or add delays
   # Configure rate limits in backend settings

Common vLLM Issues
~~~~~~~~~~~~~~~~~~

**CUDA Out of Memory**:

.. code-block:: bash

   # Check GPU memory
   nvidia-smi

   # Use smaller models or reduce batch size
   # Consider model quantization

**Model Path Errors**:

.. code-block:: bash

   # Verify absolute paths in vllm_models.yaml
   ls /absolute/path/to/model/directory

   # Ensure model files are present
   ls /path/to/model/config.json

**Import Errors**:

.. code-block:: bash

   # Install vLLM properly
   pip install vllm

   # Check CUDA compatibility
   python -c "import torch; print(torch.cuda.is_available())"

Adding New Models
-----------------

LiteLLM Models
~~~~~~~~~~~~~~

1. **Find the model identifier** from `LiteLLM documentation <https://docs.litellm.ai/docs/providers>`_

2. **Add to configuration**:

.. code-block:: yaml

   # In src/board_game_arena/configs/litellm_models.yaml
   models:
     - litellm_new_provider/new_model_name

3. **Set up API keys** in ``.env`` file if needed

4. **Test the model**:

.. code-block:: bash

   python scripts/runner.py --config configs/example_config.yaml --override \\
     agents.player_0.model=litellm_new_provider/new_model_name \\
     num_episodes=1

vLLM Models
~~~~~~~~~~~

1. **Download model files** to local directory

2. **Add model configuration**:

.. code-block:: yaml

   # In src/board_game_arena/configs/vllm_models.yaml
   models:
     - name: vllm_new_model_name
       model_path: /absolute/path/to/model
       description: Description of the new model

3. **Test the model**:

.. code-block:: bash

   python scripts/runner.py --config configs/example_config.yaml --override \\
     agents.player_0.model=vllm_new_model_name \\
     num_episodes=1

See Also
--------

* :doc:`installation` - Setting up API keys and vLLM
* :doc:`agents` - Using LLM agents in experiments
* :doc:`api_reference` - Backend implementation details
* :doc:`examples` - Backend usage examples
* `LiteLLM Documentation <https://docs.litellm.ai/>`_
* `vLLM Documentation <https://docs.vllm.ai/>`_
