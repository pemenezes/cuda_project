"""Fine-tuning QLoRA do assistente acadêmico sintético."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_LENGTH = 512
SEED = 42


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Este projeto exige treinamento em GPU CUDA.")
    if torch.cuda.get_device_properties(0).major < 7:
        raise RuntimeError("A GPU não oferece a capacidade CUDA mínima esperada.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        default="baseline_r8_lr2e4",
        help="Nome definido em configs/experiments.json.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Executa poucos passos para validar o pipeline.",
    )
    return parser.parse_args()


def tokenize_example(example: dict, tokenizer: AutoTokenizer) -> dict:
    messages = [
        {"role": "system", "content": example["system"]},
        {"role": "user", "content": example["question"]},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(
        example["answer"] + tokenizer.eos_token, add_special_tokens=False
    )["input_ids"]

    input_ids = (prompt_ids + answer_ids)[:MAX_LENGTH]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def main() -> None:
    args = parse_args()
    require_cuda()
    set_seed(SEED)

    experiments = json.loads(
        (ROOT / "configs" / "experiments.json").read_text(encoding="utf-8")
    )
    if args.experiment not in experiments:
        raise ValueError(f"Experimento desconhecido: {args.experiment}")
    config = experiments[args.experiment]
    run_name = f"{args.experiment}{'-smoke' if args.smoke_test else ''}"
    output_dir = ROOT / "artifacts" / run_name

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization,
        device_map={"": 0},
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )
    model.print_trainable_parameters()

    files = {
        "train": str(ROOT / "data" / "processed" / "train.jsonl"),
        "validation": str(ROOT / "data" / "processed" / "validation.jsonl"),
    }
    dataset = load_dataset("json", data_files=files)
    tokenized = dataset.map(
        lambda row: tokenize_example(row, tokenizer),
        remove_columns=dataset["train"].column_names,
        desc="Tokenizando",
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=run_name,
        num_train_epochs=config["epochs"],
        max_steps=5 if args.smoke_test else -1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=config["learning_rate"],
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        logging_steps=1,
        eval_strategy="steps" if args.smoke_test else "epoch",
        eval_steps=5 if args.smoke_test else None,
        save_strategy="no" if args.smoke_test else "epoch",
        save_total_limit=1,
        report_to="none",
        seed=SEED,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
        ),
    )

    torch.cuda.reset_peak_memory_stats()
    started_at = time.perf_counter()
    train_result = trainer.train()
    elapsed = time.perf_counter() - started_at
    eval_result = trainer.evaluate()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    summary = {
        "experiment": run_name,
        "model": MODEL_ID,
        "gpu": torch.cuda.get_device_name(0),
        "parameters": config,
        "train_loss": train_result.training_loss,
        "eval_loss": eval_result["eval_loss"],
        "elapsed_seconds": elapsed,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "seed": SEED,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
