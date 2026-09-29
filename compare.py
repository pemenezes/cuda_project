"""Compara o modelo base e o Barista ajustado usando o mesmo prompt."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM_PROMPT = (
    "Você é o Barista de Apartamento, um assistente para iniciantes prepararem "
    "cafés e bebidas simples em casa. Dê instruções curtas, práticas e seguras. "
    "Não invente medidas. Se a pergunta estiver fora do seu conhecimento, diga "
    "isso claramente. Não ofereça diagnóstico médico."
)


def generate(model, tokenizer, question: str) -> tuple[str, float]:
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    started_at = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - started_at
    answer = tokenizer.decode(
        output[0, inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    ).strip()
    return answer, elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", "-q", required=True)
    parser.add_argument(
        "--adapter",
        type=Path,
        default=Path("artifacts/barista_final_r16_8epochs"),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A comparação deve ser executada em GPU CUDA.")
    if not args.adapter.exists():
        raise FileNotFoundError(f"Adaptador não encontrado: {args.adapter}")

    tokenizer = AutoTokenizer.from_pretrained(str(args.adapter))
    tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        ),
        device_map={"": 0},
        dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base, str(args.adapter))
    model.eval()

    with model.disable_adapter():
        base_answer, base_time = generate(model, tokenizer, args.question)
    tuned_answer, tuned_time = generate(model, tokenizer, args.question)

    print(f"\nPERGUNTA\n{args.question}")
    print(f"\nMODELO BASE ({base_time:.2f}s)\n{base_answer}")
    print(f"\nMODELO COM FINE-TUNING ({tuned_time:.2f}s)\n{tuned_answer}")
    print(f"\nGPU\n{torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
