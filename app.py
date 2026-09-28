"""Frontend Gradio do assistente acadêmico demonstrativo."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM_PROMPT = (
    "Você é o assistente acadêmico da Faculdade Horizonte, uma instituição "
    "fictícia criada para demonstração. Responda somente com base nas regras "
    "fornecidas no treinamento. Se não souber, diga que não possui essa "
    "informação e recomende procurar a secretaria."
)


def message_text(content) -> str:
    """Converte os diferentes formatos de mensagem do Gradio em texto."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        for key in ("text", "content", "value"):
            if key in content:
                return message_text(content[key])
        return ""
    if isinstance(content, list):
        return "\n".join(
            text for part in content if (text := message_text(part))
        )
    return str(content) if content is not None else ""


def load_model(adapter_path: Path):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA não está disponível; a demonstração exige GPU.")
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adaptador não encontrado: {adapter_path}")

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
    tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        ),
        device_map={"": 0},
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()
    return model, tokenizer


def create_app(model, tokenizer, adapter_path: Path) -> gr.Blocks:
    def answer(message, history, temperature, top_p, max_tokens):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for item in history[-4:]:
            if isinstance(item, dict):
                text = message_text(item.get("content"))
                if text and item.get("role") in {"user", "assistant"}:
                    messages.append({"role": item["role"], "content": text})
            else:
                messages.extend(
                    [
                        {"role": "user", "content": message_text(item[0])},
                        {"role": "assistant", "content": message_text(item[1])},
                    ]
                )
        messages.append({"role": "user", "content": message_text(message)})
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        torch.cuda.reset_peak_memory_stats()
        started_at = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=int(max_tokens),
                do_sample=temperature > 0,
                temperature=max(float(temperature), 0.01),
                top_p=float(top_p),
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(
            generated[0, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        ).strip()
        elapsed = time.perf_counter() - started_at
        peak = torch.cuda.max_memory_allocated() / 1024**3
        return f"{text}\n\n_Tempo: {elapsed:.2f}s | Pico de VRAM: {peak:.2f} GiB_"

    with gr.Blocks(title="Assistente acadêmico — demonstração") as demo:
        gr.Markdown(
            "# Assistente acadêmico da Faculdade Horizonte\n"
            "**Projeto demonstrativo:** a instituição e todas as regras são "
            "fictícias. Não use as respostas para decisões acadêmicas reais.\n\n"
            f"Adaptador: `{adapter_path.name}` | GPU: `{torch.cuda.get_device_name(0)}`"
        )
        gr.ChatInterface(
            fn=answer,
            additional_inputs=[
                gr.Slider(0, 1.5, value=0.2, step=0.1, label="Temperatura"),
                gr.Slider(0.1, 1, value=0.9, step=0.05, label="Top-p"),
                gr.Slider(32, 256, value=128, step=16, label="Máximo de tokens"),
            ],
            examples=[
                ["Quantas horas de estágio obrigatório preciso cumprir?", 0.2, 0.9, 128],
                ["Como solicito segunda chamada?", 0.2, 0.9, 128],
                ["Como consigo uma declaração de matrícula?", 0.2, 0.9, 128],
            ],
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter",
        type=Path,
        default=Path("artifacts/final_r16_10epochs"),
    )
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    model, tokenizer = load_model(args.adapter)
    create_app(model, tokenizer, args.adapter).launch(share=args.share)


if __name__ == "__main__":
    main()
