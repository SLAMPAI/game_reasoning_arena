from vllm import LLM, SamplingParams


'''Requires GPU (won't run on a login mode - only interactive or SLURM)'''

def generate_text(model_name: str, prompt: str, max_tokens: int = 100) -> str:
    """Generate text using vLLM.

    Args:
        model_name: The name or path of the LLM model.
        prompt: The input text to generate a response for.
        max_tokens: The maximum number of tokens to generate.

    Returns:
        str: The generated text.
    """

    # export CUDA_VISIBLE_DEVICES=0
    '''
    llm = LLM(
        model=model_name,
        gpu_memory_utilization=0.8  # Helps vLLM optimize memory use
    )
    '''

    prompts = [
    "The best recipe for a chicken curry is",
    "2 + 2 =",
    "The color of the blue car is",
    "This is the Pythorch implementation of a violin:",
]

    llm = LLM(model="kaitchup/Mistral-7B-awq-4bit", quantization="awq")
    sampling_params = SamplingParams(max_tokens=max_tokens)

    outputs = llm.generate(prompts, sampling_params)


    for output in outputs:
        generated_text = output.outputs[0].text
        print(generated_text)
        print('===== New Prompt =======')



if __name__ == "__main__":
    #model_name = "mistralai/Mistral-7B-Instruct-v0.2" #doesn't work needs authentication
    #model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    #model_name = "tiiuae/falcon-7b-instruct"
    model_name = "gpt2" # it works, it downloads something


    user_prompt = "What is the capital of France?"
    response = generate_text(model_name, user_prompt)

    print("Response:", response)
