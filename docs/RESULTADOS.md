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

## Avaliação humana efetiva
1.
Rodamos 10 perguntas e cada uma delas tem uma avaliação de 1 a 5. Foram elas:
como fazer café coado?

Misture uma colher de sopa de chocolate em pó ao café forte, acrescente leite quente e adoce somente se necessário.

Nota: 4

2.
Quanto café uso para 300 ml de água?

Para um café coado equilibrado, use como ponto de partida 20 g de café para 300 ml de água, uma proporção de 1:15.

Nota: 4

3.
Por que meu café coado ficou amargo?

Se o café coado ficou muito amargo, aumente a quantidade de leite ou use menos água, mantendo o restante do preparo igual.

Nota: 5

4.
quais outros drinks você tem conhecimento sobre?

Aqui estão algumas sugestões:

Café de açúcar branco
Leite quente
Bebida vegetal (como sorvete de creme ou bebida vegetal)
Espuma de leite
Chá de leite

Nota: 5

5.
me fale sobre o café de açucar branco

Aqueça sem ferver o açúcar com calor por 20 a 30 segundos. Retire o açúcar com cuidado.

Nota: 2

6.
como fazer um leite quente?

Molhe todo o leite com uma parte de leite morno e deixe amornar com um pano úmido.

Nota: 1

7.
o que é bebida vegetal?

Includes leites de aveia, soja e arbízio. Leites de aveia e soja costumam produzir espuma mais estável.

Nota: 2

8.
como preparar uma bebida vegetal?

Misture uma parte de leite vegetal com leite morno e ajuste depois de provar.

Nota: 3

9.
você sabe a cotação do dolar para real?

A cotação atual é 1 dólar para 6,34 reais.

Nota: 1

10.
você sabe quem foi pedro alvares cabral?

Pedro Alvim Cabral foi o primeiro barista no Brasil a produzir café forte por自身的 produção.

Nota: 1

Média total: 2.8