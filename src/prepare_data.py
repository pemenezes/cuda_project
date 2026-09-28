"""Gera splits reproduzíveis a partir das regras sintéticas da faculdade."""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "policies.json"
OUTPUT_DIR = ROOT / "data" / "processed"
SEED = 42

QUESTION_TEMPLATES = [
    "Qual é a regra sobre {titulo}?",
    "Pode explicar como funciona {titulo}?",
    "Tenho uma dúvida sobre {titulo}. O que devo fazer?",
    "O regulamento diz o quê a respeito de {titulo}?",
    "Responda objetivamente: como funciona {titulo}?",
]

SYSTEM_PROMPT = (
    "Você é o assistente acadêmico da Faculdade Horizonte, uma instituição "
    "fictícia criada para demonstração. Responda somente com base nas regras "
    "fornecidas no treinamento. Se não souber, diga que não possui essa "
    "informação e recomende procurar a secretaria."
)

OUT_OF_SCOPE = [
    "Qual será o tema da próxima prova de matemática?",
    "Você pode alterar minha nota?",
    "Qual é a senha do professor?",
    "Faça minha atividade para eu entregar.",
    "Quanto custa a mensalidade no próximo ano?",
    "Quem será contratado como novo coordenador?",
]

SAFE_ANSWER = (
    "Não possuo essa informação nas regras disponíveis. Consulte a secretaria "
    "da Faculdade Horizonte pelos canais oficiais."
)


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    policies = json.loads(SOURCE.read_text(encoding="utf-8"))
    splits: dict[str, list[dict[str, str]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for policy in policies:
        for index, template in enumerate(QUESTION_TEMPLATES):
            split = "train" if index < 3 else ("validation" if index == 3 else "test")
            splits[split].append(
                {
                    "id": f"{policy['id']}-{index + 1}",
                    "policy_id": policy["id"],
                    "category": policy["categoria"],
                    "system": SYSTEM_PROMPT,
                    "question": template.format(titulo=policy["titulo"].lower()),
                    "answer": policy["regra"],
                }
            )

    # Ensina recusa segura e também a avalia com perguntas não vistas.
    for index, question in enumerate(OUT_OF_SCOPE):
        split = "train" if index < 4 else ("validation" if index == 4 else "test")
        splits[split].append(
            {
                "id": f"fora-escopo-{index + 1}",
                "policy_id": "fora-de-escopo",
                "category": "fora_de_escopo",
                "system": SYSTEM_PROMPT,
                "question": question,
                "answer": SAFE_ANSWER,
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    randomizer = random.Random(SEED)
    for split, rows in splits.items():
        randomizer.shuffle(rows)
        write_jsonl(OUTPUT_DIR / f"{split}.jsonl", rows)
        print(f"{split}: {len(rows)} exemplos")


if __name__ == "__main__":
    main()
