# Roteiro da apresentação

## 1. Problema

Assistentes gerais nem sempre fornecem medidas consistentes para preparos
caseiros. O objetivo foi adaptar uma LLM pequena para atuar como Barista de
Apartamento, com respostas curtas e verificáveis.

## 2. Escolhas técnicas

- Qwen2.5-0.5B-Instruct;
- fine-tuning QLoRA em 4 bits;
- RTX 2060 com 6 GB;
- dataset sintético e versionado;
- Gradio para demonstração.

Explique que fine-tuning ajusta uma LLM existente. Não afirme que o modelo foi
treinado do zero.

## 3. Demonstração da GPU

Mostre no terminal integrado do VSCode:

1. `nvidia-smi`;
2. `python scripts/check_gpu.py`;
3. uso de VRAM durante um treino.

## 4. Dataset

Apresente os 30 conhecimentos canônicos, cinco paráfrases por item, exemplos de
recusa e 12 diálogos com histórico. Explique a divisão entre treino, validação e
teste.

## 5. Experimentos

Compare as três configurações. Mostre loss, tempo, VRAM, métricas e duas ou três
respostas do mesmo prompt antes/depois.

## 6. Frontend

Faça perguntas sobre café coado, cappuccino e correção de amargor. Em seguida,
faça uma pergunta consecutiva e uma pergunta fora do escopo.

## 7. Conclusão

Apresente a melhor configuração, os erros nos diálogos reservados e a evolução
necessária para uso real: mais conversas, fontes revisadas, RAG com citações e
revisão humana.
