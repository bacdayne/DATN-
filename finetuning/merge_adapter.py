import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL  = "Qwen/Qwen2-1.5B-Instruct"
ADAPTER_DIR = "finetuning/qwen2-cord-lora"
MERGED_DIR  = "finetuning/qwen2-cord-merged"


def main():
    print("Loading base model on CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="cpu",
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)

    print("Merging weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model → {MERGED_DIR}")
    model.save_pretrained(MERGED_DIR, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, trust_remote_code=True)
    tokenizer.save_pretrained(MERGED_DIR)

    print("Done. Load với vLLM bằng model path:", MERGED_DIR)


if __name__ == "__main__":
    main()
