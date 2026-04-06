"""
train/stage1_train.py
Stage 1: QLoRA fine-tuning of the student model on Alpaca data.

Run on HPC:
    python train/stage1_train.py
"""
import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    STUDENT_MODEL, DATA_DIR, STAGE1_OUTPUT_DIR, LOGS_DIR,
    STAGE1_EPOCHS, STAGE1_LR, STAGE1_BATCH_SIZE, STAGE1_GRAD_ACCUM,
    STAGE1_MAX_SEQ_LEN, STAGE1_WARMUP_RATIO, STAGE1_WEIGHT_DECAY,
    LORA_RANK, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
    LOAD_IN_4BIT, BNB_4BIT_COMPUTE_DTYPE, BNB_4BIT_QUANT_TYPE, BNB_DOUBLE_QUANT,
    SEED,
)

set_seed(SEED)

ALPACA_PROMPT = (
    "Below is an instruction that describes a task, paired with an input that provides "
    "further context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)

ALPACA_PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{output}"
)


def format_example(ex: dict) -> str:
    if ex.get("input", "").strip():
        return ALPACA_PROMPT.format(
            instruction=ex["instruction"],
            input=ex["input"],
            output=ex["output"],
        )
    return ALPACA_PROMPT_NO_INPUT.format(
        instruction=ex["instruction"],
        output=ex["output"],
    )


def load_data() -> Dataset:
    train_path = os.path.join(DATA_DIR, "alpaca_train.json")
    data = json.loads(Path(train_path).read_text(encoding="utf-8"))
    formatted = [{"text": format_example(ex)} for ex in data]
    return Dataset.from_list(formatted)


def load_model_and_tokenizer():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=LOAD_IN_4BIT,
        bnb_4bit_compute_dtype=getattr(torch, BNB_4BIT_COMPUTE_DTYPE),
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_use_double_quant=BNB_DOUBLE_QUANT,
    )

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model = prepare_model_for_kbit_training(model)

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
    print(f"=== Stage 1: Alpaca Fine-Tuning ===")
    print(f"Student model: {STUDENT_MODEL}")
    print(f"Output dir:    {STAGE1_OUTPUT_DIR}")

    os.makedirs(STAGE1_OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    dataset = load_data()
    print(f"Training examples: {len(dataset)}")

    model, tokenizer = load_model_and_tokenizer()

    training_args = TrainingArguments(
        output_dir=STAGE1_OUTPUT_DIR,
        num_train_epochs=STAGE1_EPOCHS,
        per_device_train_batch_size=STAGE1_BATCH_SIZE,
        gradient_accumulation_steps=STAGE1_GRAD_ACCUM,
        learning_rate=STAGE1_LR,
        warmup_ratio=STAGE1_WARMUP_RATIO,
        weight_decay=STAGE1_WEIGHT_DECAY,
        fp16=True,
        logging_dir=os.path.join(LOGS_DIR, "stage1"),
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
        max_seq_length=STAGE1_MAX_SEQ_LEN,
        packing=False,
    )

    print("Starting Stage 1 training...")
    trainer.train()
    trainer.save_model(STAGE1_OUTPUT_DIR)
    tokenizer.save_pretrained(STAGE1_OUTPUT_DIR)
    print(f"Stage 1 complete. Adapter saved to: {STAGE1_OUTPUT_DIR}")


if __name__ == "__main__":
    train()
