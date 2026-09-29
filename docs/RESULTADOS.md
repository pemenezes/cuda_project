# Resultados — Barista de Apartamento

Resultados obtidos localmente em 28/09/2026. As execuções completas estão em
`results/experiments.csv` e os exemplos em `results/evaluation_*.json`.

## Ambiente

- GPU: NVIDIA GeForce RTX 2060, 6 GB
- Driver: 616.64
- Versão CUDA do PyTorch: 13.0
- Versão do PyTorch: 2.14.0+cu130
- Data dos experimentos: 28/09/2026

## Comparação

- Modelo base: ROUGE-L 0,126 e F1 0,159.
- Rank 8, learning rate 0,0002, 2 épocas: loss de validação 1,172,
  ROUGE-L 0,222 e F1 0,298.
- Rank 8, learning rate 0,0001, 2 épocas: loss de validação 1,777,
  ROUGE-L 0,172 e F1 0,217.
- Rank 16, learning rate 0,0002, 2 épocas: loss de validação 0,717,
  ROUGE-L 0,255 e F1 0,314.
- Rank 16, learning rate 0,0002, 8 épocas: loss de validação 0,079,
  ROUGE-L 0,927 e F1 0,930.

A configuração final levou 648,8 segundos e usou pico de 1,35 GiB de VRAM. Em
testes adicionais fora das métricas, respondeu corretamente perguntas naturais
sobre cappuccino, amargor, proporção e uma continuação sobre medir sem balança.
No conjunto formal, a pergunta sobre coquetel alcoólico e os dois diálogos
reservados tiveram F1 abaixo de 0,9, mostrando que manter contexto e recusar
pedidos ainda é mais difícil que responder perguntas isoladas.

## Avaliação humana cega

Para cada resposta do conjunto de teste, atribua de 1 a 5:

1. correção: respeita a regra canônica;
2. clareza: responde de forma direta e compreensível;
3. segurança: evita inventar informações e recusa pedidos fora do escopo.

Embaralhe a identificação dos modelos durante a avaliação. Depois revele os
nomes e justifique a configuração escolhida.

## Análise

O aumento do rank de 8 para 16 melhorou loss e métricas em duas épocas. Aumentar
o treinamento para 8 épocas foi decisivo para memorizar as orientações.

O resultado não demonstra generalização para receitas novas. O teste usa uma
paráfrase reservada dos mesmos fatos vistos no treino, e o desempenho quase
perfeito nas perguntas isoladas indica forte memorização. Nos diálogos, o
modelo errou uma recomendação sobre moagem e uma quantidade de leite. Uma
próxima versão deve ampliar conversas de treinamento e usar RAG com fontes
revisadas.
