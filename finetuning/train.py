import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import LoraConfig, TaskType
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

MODEL_NAME  = "Qwen/Qwen2-1.5B-Instruct"
TRAIN_FILE  = "finetuning/data/train.jsonl"
VAL_FILE    = "finetuning/data/val.jsonl"
OUTPUT_DIR  = "finetuning/checkpoints"
ADAPTER_DIR = "finetuning/qwen2-cord-lora"
MAX_SEQ_LEN = 1024


def load_jsonl(path):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def main():
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Format prompt+completion thành ChatML của Qwen2
    def format_samples(examples):
        texts = []
        for prompt, completion in zip(examples["prompt"], examples["completion"]):
            messages = [
                {"role": "user",      "content": prompt},
                {"role": "assistant", "content": completion},
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
        return {"text": texts}

    train_raw = load_jsonl(TRAIN_FILE)
    val_raw   = load_jsonl(VAL_FILE)

    train_dataset = Dataset.from_list(train_raw).map(format_samples, batched=True)
    val_dataset   = Dataset.from_list(val_raw).map(format_samples, batched=True)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    print("Sample (first 300 chars):")
    print(train_dataset[0]["text"][:300])
    print()

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.enable_input_require_grads()

    # LoRA — chỉ train ~3M / 1.5B params
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )

    # Chỉ tính loss trên phần assistant (completion), không tính trên prompt
    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        bf16=True,
        optim="adamw_torch",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        data_collator=collator,
        peft_config=lora_config,
        tokenizer=tokenizer,
    )

    trainer.train()

    trainer.model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print(f"\nAdapter saved → {ADAPTER_DIR}")


if __name__ == "__main__":
    main()
