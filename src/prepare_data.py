"""Gera splits reproduzíveis do dataset Barista de Apartamento."""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "barista_knowledge.json"
OUTPUT_DIR = ROOT / "data" / "processed"
SEED = 42

QUESTION_TEMPLATES = [
    "Como faço {titulo}?",
    "Pode me explicar {titulo}?",
    "Qual é a orientação para {titulo}?",
    "Sou iniciante: o que preciso saber sobre {titulo}?",
    "Responda de forma direta sobre {titulo}.",
]

SYSTEM_PROMPT = (
    "Você é o Barista de Apartamento, um assistente para iniciantes prepararem "
    "cafés e bebidas simples em casa. Dê instruções curtas, práticas e seguras. "
    "Não invente medidas. Se a pergunta estiver fora do seu conhecimento, diga "
    "isso claramente. Não ofereça diagnóstico médico."
)

OUT_OF_SCOPE = [
    "Como faço uma lasanha completa?",
    "Qual ação devo comprar hoje?",
    "Você pode diagnosticar minha dor de cabeça?",
    "Escreva meu trabalho da faculdade.",
    "Como conserto a instalação elétrica da cozinha?",
    "Qual remédio devo tomar para dormir?",
    "Quem vai ganhar o campeonato?",
    "Ensine uma receita de coquetel alcoólico.",
]

SAFE_ANSWER = (
    "Isso está fora do escopo do Barista de Apartamento. Posso ajudar com café "
    "e bebidas caseiras simples dentro das orientações disponíveis."
)

NATURAL_QUESTIONS = {
    "coado-proporcao": "Quanto café devo usar para 300 ml de água?",
    "coado-temperatura": "Qual deve ser a temperatura da água do café coado?",
    "coado-moagem": "Qual moagem devo usar no café coado?",
    "coado-escalda": "Preciso molhar o filtro de papel antes de fazer café?",
    "coado-floracao": "O que é a pré-infusão do café coado?",
    "prensa-receita": "Como preparo café na prensa francesa?",
    "italiana-preparo": "Como preparo café na cafeteira italiana?",
    "italiana-retirar": "Como evito que o café da cafeteira italiana fique amargo?",
    "soluvel-proporcao": "Quanto café solúvel devo usar por xícara?",
    "coldbrew-preparo": "Como preparo cold brew em casa?",
    "coldbrew-servir": "Como devo diluir o concentrado de cold brew?",
    "cappuccino-caseiro": "Como faço cappuccino sem máquina?",
    "latte-caseiro": "Como faço um latte caseiro?",
    "mocha-caseiro": "Como faço mocha em casa?",
    "cafe-gelado": "Como faço café gelado sem ficar aguado?",
    "affogato": "Como faço um affogato simples?",
    "sem-cafeina": "Que bebida cremosa posso fazer sem cafeína?",
    "espuma-pote": "Como faço espuma de leite usando um pote?",
    "espuma-prensa": "Como faço espuma de leite na prensa francesa?",
    "leite-vegetal": "Qual leite vegetal funciona melhor no café?",
    "sem-balanca": "Como meço o café se não tenho balança?",
    "amargo": "Por que meu café ficou amargo e como posso corrigir?",
    "azedo": "Por que meu café ficou ácido ou azedo?",
    "fraco": "Como posso corrigir um café que ficou fraco?",
    "forte": "Como posso corrigir um café que ficou forte demais?",
    "graos-armazenamento": "Qual é a melhor forma de guardar o café?",
    "cafe-pronto": "Posso guardar café depois de pronto?",
    "coldbrew-seguranca": "Como armazeno cold brew com segurança?",
    "leite-seguranca": "Posso deixar uma bebida com leite fora da geladeira?",
    "cafeina-limite": "O que faço se sou sensível à cafeína?",
}

DIALOGUES = [
    {
        "history": [
            {"role": "user", "content": "Meu café coado ficou muito amargo."},
            {
                "role": "assistant",
                "content": "Tente moagem mais grossa, água menos quente ou menor tempo de contato.",
            },
        ],
        "question": "Qual dessas mudanças você recomenda testar primeiro?",
        "answer": "Teste uma mudança por vez. Comece deixando a moagem um pouco mais grossa e mantenha as outras medidas iguais.",
    },
    {
        "history": [
            {"role": "user", "content": "Quero preparar 300 ml de café coado."},
            {
                "role": "assistant",
                "content": "Use como ponto de partida 20 g de café para 300 ml de água.",
            },
        ],
        "question": "E se eu não tiver balança?",
        "answer": "Use aproximadamente duas colheres de sopa rasas de café moído para 300 ml de água e ajuste depois.",
    },
    {
        "history": [
            {"role": "user", "content": "Como preparo cold brew?"},
            {
                "role": "assistant",
                "content": "Misture café grosso e água fria na proporção 1:8 e deixe na geladeira por 12 a 16 horas.",
            },
        ],
        "question": "Posso beber esse concentrado puro?",
        "answer": "Pode, mas ele é concentrado. O ponto de partida é misturar uma parte de cold brew com uma parte de água ou leite.",
    },
    {
        "history": [
            {"role": "user", "content": "Quero um cappuccino sem máquina."},
            {
                "role": "assistant",
                "content": "Use café forte, leite quente e espuma em partes iguais.",
            },
        ],
        "question": "Como faço a espuma?",
        "answer": "Agite leite morno em um pote fechado por 30 a 45 segundos, preenchendo no máximo metade do pote.",
    },
    {
        "history": [
            {"role": "user", "content": "Meu café ficou muito fraco."},
            {
                "role": "assistant",
                "content": "Aumente a dose de café ou diminua a quantidade de água.",
            },
        ],
        "question": "Devo mudar as duas coisas ao mesmo tempo?",
        "answer": "Não. Altere uma variável por vez para entender o efeito. Comece aumentando um pouco a quantidade de café.",
    },
    {
        "history": [
            {"role": "user", "content": "Posso usar leite vegetal no latte?"},
            {
                "role": "assistant",
                "content": "Sim. Leites de aveia e soja costumam produzir espuma mais estável.",
            },
        ],
        "question": "Devo deixar ferver?",
        "answer": "Não. Aqueça o leite vegetal sem deixar ferver para preservar sabor e textura.",
    },
    {
        "history": [
            {"role": "user", "content": "Como guardo o café em pó?"},
            {
                "role": "assistant",
                "content": "Use recipiente bem fechado, longe de luz, calor, umidade e odores.",
            },
        ],
        "question": "Posso deixar ao lado do fogão?",
        "answer": "Não é recomendado, porque o fogão produz calor. Guarde o recipiente em um armário fresco e seco.",
    },
    {
        "history": [
            {"role": "user", "content": "Fiz café gelado com leite."},
            {
                "role": "assistant",
                "content": "Sirva logo e mantenha os ingredientes refrigerados.",
            },
        ],
        "question": "Posso deixar na mesa a tarde toda?",
        "answer": "Não. Bebidas com leite não devem ficar por longos períodos fora da geladeira. Em caso de dúvida, descarte.",
    },
    {
        "history": [
            {"role": "user", "content": "Minha cafeteira italiana começou a borbulhar."},
            {
                "role": "assistant",
                "content": "Isso indica que a extração está terminando.",
            },
        ],
        "question": "O que faço agora?",
        "answer": "Retire a cafeteira do fogo e resfrie a base com um pano úmido para interromper a extração.",
    },
    {
        "history": [
            {"role": "user", "content": "Quero fazer mocha."},
            {
                "role": "assistant",
                "content": "Misture chocolate em pó ao café forte e acrescente leite quente.",
            },
        ],
        "question": "Quanto chocolate eu uso?",
        "answer": "Comece com uma colher de sopa de chocolate em pó e ajuste depois de provar.",
    },
    {
        "history": [
            {"role": "user", "content": "Meu café ficou muito ácido."},
            {
                "role": "assistant",
                "content": "Tente moagem mais fina, água mais quente ou maior tempo de contato.",
            },
        ],
        "question": "Posso começar pela moagem?",
        "answer": "Sim. Ajuste a moagem um pouco mais fina e mantenha as outras variáveis iguais para comparar.",
    },
    {
        "history": [
            {"role": "user", "content": "Quero uma bebida sem cafeína."},
            {
                "role": "assistant",
                "content": "Você pode aquecer leite ou bebida vegetal com cacau e canela.",
            },
        ],
        "question": "Qual quantidade de leite?",
        "answer": "Use cerca de 200 ml de leite ou bebida vegetal e ajuste cacau, canela e açúcar depois de provar.",
    },
]


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    knowledge = json.loads(SOURCE.read_text(encoding="utf-8"))
    splits: dict[str, list[dict[str, str]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for item in knowledge:
        for index, template in enumerate(QUESTION_TEMPLATES):
            split = "train" if index < 3 else ("validation" if index == 3 else "test")
            splits[split].append(
                {
                    "id": f"{item['id']}-{index + 1}",
                    "knowledge_id": item["id"],
                    "category": item["categoria"],
                    "system": SYSTEM_PROMPT,
                    "history": [],
                    "question": template.format(titulo=item["titulo"].lower()),
                    "answer": item["regra"],
                }
            )
        splits["train"].append(
            {
                "id": f"{item['id']}-natural",
                "knowledge_id": item["id"],
                "category": item["categoria"],
                "system": SYSTEM_PROMPT,
                "history": [],
                "question": NATURAL_QUESTIONS[item["id"]],
                "answer": item["regra"],
            }
        )

    # Ensina recusa segura e também a avalia com perguntas não vistas.
    for index, question in enumerate(OUT_OF_SCOPE):
        split = "train" if index < 6 else ("validation" if index == 6 else "test")
        splits[split].append(
            {
                "id": f"fora-escopo-{index + 1}",
                "knowledge_id": "fora-de-escopo",
                "category": "fora_de_escopo",
                "system": SYSTEM_PROMPT,
                "history": [],
                "question": question,
                "answer": SAFE_ANSWER,
            }
        )

    for index, dialogue in enumerate(DIALOGUES):
        split = "train" if index < 8 else ("validation" if index < 10 else "test")
        splits[split].append(
            {
                "id": f"dialogo-{index + 1}",
                "knowledge_id": "dialogo",
                "category": "dialogo",
                "system": SYSTEM_PROMPT,
                **dialogue,
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
