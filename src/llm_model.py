from transformers import (
    AutoTokenizer, 
    AutoConfig,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline as transformers_pipeline
)
import torch

from langchain.llms.huggingface_pipeline import (
    HuggingFacePipeline
) 

def load_4bit_mistral_7b_instruct_v01():
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    model_config = AutoConfig.from_pretrained(model_name)
    quantized_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, 
        bnb_4bit_use_double_quant=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        quantization_config=quantized_config
        )

    return tokenizer, model

def setup_hf_pipeline(tokenizer, model):

    new_transformer_pipeline = transformers_pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        framework="pt",
        temperature=0.2,
        repetition_penalty=1.1, 
        return_full_text=True, 
        max_new_tokens=1000,
        do_sample=True
    )

    hf_pipeline = HuggingFacePipeline(
        pipeline=new_transformer_pipeline
    )
    return hf_pipeline

if __name__ == "__main__":
    import gc
    torch.cuda.empty_cache()
    gc.collect()
    tokenizer, model = load_4bit_mistral_7b_instruct_v01()
    print("Model and tokenizer loaded successfully")
    generator = setup_hf_pipeline(tokenizer, model)
    result = generator.invoke("Where is Singapore located?")       
    print(result)
    # printed response:
    """
    A: a country in Southeast Asia
    b a city-state on the island of Singapore
    c a country in South America
    d a country in Europe
    e a country in Africa
    ```

    The answer is `b`, because it's a city-state on the island of Singapore.
    """