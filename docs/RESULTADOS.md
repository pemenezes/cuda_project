# Resultados dos experimentos

Resultados obtidos localmente em 28/09/2026. As execuções completas estão em
`results/experiments.csv` e os exemplos em `results/evaluation_*.json`.

## Ambiente

- GPU: NVIDIA GeForce RTX 2060, 6 GB
- Driver: 616.64
- Versão CUDA do PyTorch: 13.0
- Versão do PyTorch: 2.14.0+cu130
- Data dos experimentos: 28/09/2026

## Comparação

- Modelo base: ROUGE-L 0,112 e F1 0,135.
- Rank 8, learning rate 0,0002, 2 épocas: loss de validação 1,235,
  ROUGE-L 0,244 e F1 0,252.
- Rank 8, learning rate 0,0001, 2 épocas: loss de validação 1,653,
  ROUGE-L 0,245 e F1 0,258.
- Rank 16, learning rate 0,0002, 2 épocas: loss de validação 0,820,
  ROUGE-L 0,284 e F1 0,278.
- Rank 16, learning rate 0,0002, 10 épocas: loss de validação 0,0077,
  ROUGE-L 0,963 e F1 0,960.

A configuração final levou 443,9 segundos e usou pico de 1,29 GiB de VRAM. Os
24 testes sobre regras foram respondidos exatamente. A única pergunta fora do
escopo falhou: o modelo inventou quem escolheria um novo coordenador. Esse erro
deve aparecer na apresentação, pois mostra que loss e média altas não garantem
segurança.

## Avaliação humana cega

Para cada resposta do conjunto de teste, atribua de 1 a 5:

1. correção: respeita a regra canônica;
2. clareza: responde de forma direta e compreensível;
3. segurança: evita inventar informações e recusa pedidos fora do escopo.

Embaralhe a identificação dos modelos durante a avaliação. Depois revele os
nomes e justifique a configuração escolhida.

## Análise

O aumento do rank de 8 para 16 melhorou loss e métricas em duas épocas, mas as
respostas ainda trocavam fatos: o modelo respondeu 120 em vez de 240 horas de
estágio. Aumentar o treinamento para 10 épocas foi decisivo para memorizar as
24 regras.

O resultado não demonstra generalização para regulamentos novos. O teste usa
uma paráfrase reservada dos mesmos fatos vistos no treino e o desempenho quase
perfeito indica forte memorização. A falha fora do escopo confirma que ainda é
necessário ampliar exemplos de recusa ou, em um produto real, usar RAG com
fontes oficiais e validação humana.
