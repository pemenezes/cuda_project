"""Avalia o modelo base ou um adaptador com o conjunto de teste reservado."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def normalize_tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


def token_f1(reference: str, prediction: str) -> float:
    reference_tokens = normalize_tokens(reference)
    prediction_tokens = normalize_tokens(prediction)
    if not reference_tokens or not prediction_tokens:
        return float(reference_tokens == prediction_tokens)
    common_count = sum(
        (Counter(reference_tokens) & Counter(prediction_tokens)).values()
    )
    precision = common_count / len(prediction_tokens)
    recall = common_count / len(reference_tokens)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, help="Pasta do adaptador LoRA.")
    parser.add_argument("--name", default="base", help="Nome usado no resultado.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A avaliação deve ser executada com CUDA.")

    tokenizer_path = str(args.adapter) if args.adapter else MODEL_ID
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        ),
        device_map={"": 0},
        torch_dtype=torch.float16,
    )
    model = (
        PeftModel.from_pretrained(base_model, str(args.adapter))
        if args.adapter
        else base_model
    )
    model.eval()

    test_path = ROOT / "data" / "processed" / "test.jsonl"
    examples = [
        json.loads(line)
        for line in test_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    results = []

    for example in examples:
        messages = [
            {"role": "system", "content": example["system"]},
            {"role": "user", "content": example["question"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        started_at = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - started_at
        prediction = tokenizer.decode(
            generated[0, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        ).strip()
        results.append(
            {
                "id": example["id"],
                "category": example["category"],
                "question": example["question"],
                "reference": example["answer"],
                "prediction": prediction,
                "rouge_l": rouge.score(example["answer"], prediction)["rougeL"].fmeasure,
                "token_f1": token_f1(example["answer"], prediction),
                "latency_seconds": elapsed,
                "human_correctness_1_to_5": None,
                "human_clarity_1_to_5": None,
                "human_safety_1_to_5": None,
            }
        )

    summary = {
        "name": args.name,
        "adapter": str(args.adapter) if args.adapter else None,
        "model": MODEL_ID,
        "gpu": torch.cuda.get_device_name(0),
        "examples": len(results),
        "mean_rouge_l": statistics.fmean(row["rouge_l"] for row in results),
        "mean_token_f1": statistics.fmean(row["token_f1"] for row in results),
        "mean_latency_seconds": statistics.fmean(
            row["latency_seconds"] for row in results
        ),
    }
    payload = {"summary": summary, "responses": results}
    output_dir = ROOT / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"evaluation_{args.name}.json"
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Resultado salvo em {output_path}")


if __name__ == "__main__":
    main()
