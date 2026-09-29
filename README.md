# Barista de Apartamento com fine-tuning de LLM

Projeto acadêmico de fine-tuning local de uma LLM usando GPU. O sistema adapta
o `Qwen/Qwen2.5-0.5B-Instruct` com QLoRA para responder perguntas sobre preparo
de cafés e bebidas simples em casa.

> O conteúdo é demonstrativo e não substitui orientação médica, nutricional ou
> de segurança alimentar.

## Integrantes

- Anny Carolina — RM98295
- Eric Carvalho — RM550249
- Henrique Lima — RM551528
- Pedro Menezes — RM97432
- Renato Izumi — RM99242

O relatório completo para entrega está em
[docs/RELATORIO_WORD.md](docs/RELATORIO_WORD.md).

## Objetivo

O projeto demonstra:

- criação controlada de um dataset de perguntas e respostas;
- fine-tuning de uma LLM existente, e não treinamento do zero;
- três experimentos controlados e uma configuração final;
- comparação objetiva entre o modelo base e os modelos ajustados;
- treinamento e teste de diálogos com perguntas consecutivas;
- treinamento, avaliação e inferência em uma NVIDIA GeForce RTX 2060;
- interface local para o usuário escrever perguntas.

## Estrutura

```text
.
├── app.py                       # frontend Gradio
├── compare.py                   # comparação antes/depois
├── configs/experiments.json     # parâmetros comparados
├── data/
│   ├── barista_knowledge.json   # conhecimentos canônicos
│   └── processed/               # splits gerados (não versionados)
├── docs/                        # relatório, resultados e apresentação
├── results/                     # avaliações e métricas leves
├── scripts/check_gpu.py         # prova de execução CUDA
├── src/
│   ├── prepare_data.py
│   ├── train.py
│   └── evaluate.py
└── tests/
    ├── test_app.py
    └── test_dataset.py
```

## 1. Ambiente no Windows

O ambiente validado utiliza Python 3.12.2 e o driver NVIDIA 616.64. A RTX 2060
tem compute capability 7.5 e 6 GB de VRAM dedicada.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Para o ambiente validado neste projeto (driver 616.64), instale o wheel oficial
CUDA 13.0 do PyTorch e depois as demais dependências:

```powershell
pip install torch==2.14.0+cu130 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
python scripts/check_gpu.py
```

Em outra máquina, confirme o comando atual em
[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).
O runtime CUDA usado pelo PyTorch vem no próprio wheel. Nesta máquina, o CUDA
Toolkit 13.0 também foi instalado em `D:\dev\cuda`, com driver 616.64. Registre
no trabalho as saídas de `nvidia-smi`, `nvcc --version` e `check_gpu.py`.

## 2. Preparar o dataset

```powershell
python src/prepare_data.py
pytest
```

São gerados 134 exemplos de treino, 33 de validação e 33 de teste. Cada
conhecimento possui paráfrases separadas por split; também há exemplos fora do
escopo e diálogos com histórico. Consulte [data/README.md](data/README.md).

## 3. Validar e treinar na GPU

Primeiro execute um teste curto:

```powershell
python src/train.py --experiment barista_r8_lr2e4 --smoke-test
```

Depois execute os três experimentos:

```powershell
python src/train.py --experiment barista_r8_lr2e4
python src/train.py --experiment barista_r8_lr1e4
python src/train.py --experiment barista_r16_lr2e4
```

Como os testes de duas épocas ainda produziram respostas imprecisas, foi
executada uma quarta configuração de refinamento:

```powershell
python src/train.py --experiment barista_final_r16_8epochs
```

Cada execução salva somente o adaptador LoRA e um `run_summary.json` em
`artifacts/`. Os experimentos comparam taxa de aprendizado e rank LoRA,
mantendo os demais fatores fixos; a rodada final testa o efeito do número de
épocas. O script registra GPU, tempo, loss e pico de VRAM.

Se houver erro de memória, feche programas que usam a GPU e reduza
`MAX_LENGTH` em `src/train.py`. Não aumente o batch por dispositivo na RTX 2060.

## 4. Avaliar

Avalie o modelo original e cada adaptador com geração determinística:

```powershell
python src/evaluate.py --name barista_base
python src/evaluate.py --adapter artifacts/barista_r8_lr2e4 --name barista_r8_lr2e4
python src/evaluate.py --adapter artifacts/barista_r8_lr1e4 --name barista_r8_lr1e4
python src/evaluate.py --adapter artifacts/barista_r16_lr2e4 --name barista_r16_lr2e4
python src/evaluate.py --adapter artifacts/barista_final_r16_8epochs --name barista_final_v2
```

Os JSONs em `results/` contêm ROUGE-L, F1 por tokens, latência, respostas lado a
lado e campos para avaliação humana de correção, clareza e segurança (1 a 5).
ROUGE e F1 ajudam na comparação, mas não substituem a revisão factual.

## 5. Abrir o frontend

Escolha o adaptador com melhor avaliação:

```powershell
python app.py --adapter artifacts/barista_final_r16_8epochs
```

O Gradio mostrará o endereço local. A interface permite controlar temperatura,
top-p e limite de tokens, e exibe latência e pico de VRAM.

## 6. Comparar antes e depois do fine-tuning

Envie o mesmo prompt ao Qwen original e ao adaptador final:

```powershell
python compare.py --question "Como faço cappuccino sem máquina?"
```

O comando usa o mesmo prompt de sistema e geração determinística nos dois casos,
imprime as respostas lado a lado e confirma a GPU utilizada.

## Evidências para a apresentação

Capture dentro do terminal integrado do VSCode:

1. `nvidia-smi`, com nome da RTX 2060, driver e uso;
2. `nvcc --version`, caso o Toolkit tenha sido instalado;
3. `python scripts/check_gpu.py`;
4. `nvidia-smi` durante o fine-tuning, mostrando o processo Python e VRAM;
5. os resumos dos três experimentos;
6. respostas do modelo base e do melhor adaptador para as mesmas perguntas;
7. o frontend Gradio em funcionamento.

Os adaptadores e caches são ignorados pelo Git por serem grandes. O GitHub deve
conter código, dataset canônico, configurações, resultados leves, screenshots e
instruções reproduzíveis.

## Limitações

- As receitas e orientações são um dataset demonstrativo.
- O dataset é pequeno e avalia principalmente memorização e paráfrase.
- Os dois diálogos reservados ainda são mais difíceis que perguntas isoladas.
- O modelo pode alucinar; em produção seria necessário recuperar os documentos
  de referência (RAG), citar fontes e manter revisão humana.
- A avaliação humana deve ser preenchida sem saber qual configuração produziu
  cada resposta, reduzindo viés na escolha do melhor experimento.
