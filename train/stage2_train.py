"""
train/stage2_train.py
Stage 2: Continue QLoRA fine-tuning from the Stage 1 checkpoint
on the teacher-generated JSON Instruct dataset.

Run on HPC:
    python train/stage2_train.py
"""
import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
)
from trl import SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    STUDENT_MODEL, DATA_DIR, STAGE1_OUTPUT_DIR, STAGE2_OUTPUT_DIR, LOGS_DIR,
    STAGE2_EPOCHS, STAGE2_LR, STAGE2_BATCH_SIZE, STAGE2_GRAD_ACCUM,
    STAGE2_MAX_SEQ_LEN, STAGE2_WARMUP_RATIO, STAGE2_WEIGHT_DECAY,
    LORA_RANK, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
    LOAD_IN_4BIT, BNB_4BIT_COMPUTE_DTYPE, BNB_4BIT_QUANT_TYPE, BNB_DOUBLE_QUANT,
    SEED,
)

set_seed(SEED)

JSON_PROMPT = (
    "Below is an instruction describing a structured-output task. "
    "Respond ONLY with valid JSON, no explanation or markdown.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)

JSON_PROMPT_NO_INPUT = (
    "Below is an instruction describing a structured-output task. "
    "Respond ONLY with valid JSON, no explanation or markdown.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{output}"
)


def format_example(ex: dict) -> str:
    if ex.get("input", "").strip():
        return JSON_PROMPT.format(
            instruction=ex["instruction"],
            input=ex["input"],
            output=ex["output"],
        )
    return JSON_PROMPT_NO_INPUT.format(
        instruction=ex["instruction"],
        output=ex["output"],
    )


def load_data() -> Dataset:
    train_path = os.path.join(DATA_DIR, "json_instruct_train.json")
    data = json.loads(Path(train_path).read_text(encoding="utf-8"))
    formatted = [{"text": format_example(ex)} for ex in data]
    return Dataset.from_list(formatted)


def load_model_and_tokenizer():
    """Load the Stage 1 checkpoint as the starting point for Stage 2."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=LOAD_IN_4BIT,
        bnb_4bit_compute_dtype=getattr(torch, BNB_4BIT_COMPUTE_DTYPE),
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_use_double_quant=BNB_DOUBLE_QUANT,
    )

    tokenizer = AutoTokenizer.from_pretrained(STAGE1_OUTPUT_DIR, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load base model with quantization
    base_model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    # Load Stage 1 LoRA adapter on top of base model
    print(f"Loading Stage 1 adapter from: {STAGE1_OUTPUT_DIR}")
    model = PeftModel.from_pretrained(base_model, STAGE1_OUTPUT_DIR)

    # Merge Stage 1 adapter into base weights, then add a NEW LoRA for Stage 2
    print("Merging Stage 1 adapter...")
    model = model.merge_and_unload()

    # Add fresh LoRA adapter for Stage 2
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def train():
    print(f"=== Stage 2: JSON Instruct Fine-Tuning ===")
    print(f"Starting from Stage 1 checkpoint: {STAGE1_OUTPUT_DIR}")
    print(f"Output dir: {STAGE2_OUTPUT_DIR}")

    if not os.path.exists(STAGE1_OUTPUT_DIR):
        raise FileNotFoundError(
            f"Stage 1 checkpoint not found at {STAGE1_OUTPUT_DIR}. "
            "Run stage1_train.py first."
        )

    os.makedirs(STAGE2_OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    dataset = load_data()
    print(f"Training examples: {len(dataset)}")

    model, tokenizer = load_model_and_tokenizer()

    training_args = TrainingArguments(
        output_dir=STAGE2_OUTPUT_DIR,
        num_train_epochs=STAGE2_EPOCHS,
        per_device_train_batch_size=STAGE2_BATCH_SIZE,
        gradient_accumulation_steps=STAGE2_GRAD_ACCUM,
        learning_rate=STAGE2_LR,          # lower LR to reduce forgetting
        warmup_ratio=STAGE2_WARMUP_RATIO,
        weight_decay=STAGE2_WEIGHT_DECAY,
        fp16=True,
        logging_dir=os.path.join(LOGS_DIR, "stage2"),
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        seed=SEED,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        dataloader_num_workers=4,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=STAGE2_MAX_SEQ_LEN,
        packing=False,
    )

    print("Starting Stage 2 training...")
    trainer.train()
    trainer.save_model(STAGE2_OUTPUT_DIR)
    tokenizer.save_pretrained(STAGE2_OUTPUT_DIR)
    print(f"Stage 2 complete. Adapter saved to: {STAGE2_OUTPUT_DIR}")


if __name__ == "__main__":
    train()
