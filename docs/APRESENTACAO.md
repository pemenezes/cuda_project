# Roteiro da apresentação

## 1. Problema

Assistentes gerais não conhecem necessariamente regulamentos internos. O
objetivo foi adaptar uma LLM pequena para responder dúvidas acadêmicas de uma
instituição fictícia, com respostas verificáveis.

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

Apresente as 24 regras canônicas, cinco paráfrases por regra e os exemplos de
recusa. Explique a divisão entre treino, validação e teste e deixe explícito que
a faculdade é fictícia.

## 5. Experimentos

Compare as três configurações. Mostre loss, tempo, VRAM, métricas e duas ou três
respostas do mesmo prompt antes/depois.

## 6. Frontend

Faça perguntas sobre estágio, atividades complementares e serviços. Em seguida
faça uma pergunta fora do escopo para demonstrar a recusa.

## 7. Conclusão

Apresente a melhor configuração, as limitações e a evolução necessária para uso
real: documentos oficiais, RAG com citações, atualização das normas e revisão
humana.
