# Assistente acadêmico com fine-tuning de LLM

Projeto acadêmico de fine-tuning local de uma LLM usando GPU. O sistema adapta
o `Qwen/Qwen2.5-0.5B-Instruct` com QLoRA para responder perguntas sobre regras e
serviços de uma faculdade.

> A **Faculdade Horizonte** e todas as suas regras são fictícias. O projeto é
> demonstrativo e não representa regulamentos da FIAP ou de outra instituição.

## Objetivo

O projeto demonstra:

- criação controlada de um dataset de perguntas e respostas;
- fine-tuning de uma LLM existente, e não treinamento do zero;
- três experimentos com parâmetros diferentes;
- comparação objetiva entre o modelo base e os modelos ajustados;
- treinamento, avaliação e inferência em uma NVIDIA GeForce RTX 2060;
- interface local para o usuário escrever perguntas.

## Estrutura

```text
.
├── app.py                       # frontend Gradio
├── configs/experiments.json     # parâmetros comparados
├── data/
│   ├── policies.json            # regras fictícias canônicas
│   └── processed/               # splits gerados (não versionados)
├── scripts/check_gpu.py         # prova de execução CUDA
├── src/
│   ├── prepare_data.py
│   ├── train.py
│   └── evaluate.py
└── tests/test_dataset.py
```

## 1. Ambiente no Windows

Recomenda-se Python 3.11 e o driver NVIDIA mais recente compatível com a RTX
2060. A placa tem compute capability 7.5 e 6 GB de VRAM dedicada.

```powershell
py -3.11 -m venv .venv
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
O runtime CUDA usado pelo PyTorch vem no próprio wheel. O CUDA Toolkit separado
só é necessário para ferramentas como `nvcc`; caso seja instalado, sua versão
deve ser suportada pelo driver mostrado em `nvidia-smi`. Registre no trabalho
as saídas de `nvidia-smi`, `nvcc --version` e `check_gpu.py`.

## 2. Preparar o dataset

```powershell
python src/prepare_data.py
pytest
```

São gerados 76 exemplos de treino, 25 de validação e 25 de teste. Cada regra
possui paráfrases separadas por split. Consulte [data/README.md](data/README.md)
para entender o limite metodológico dessa estratégia.

## 3. Validar e treinar na GPU

Primeiro execute um teste curto:

```powershell
python src/train.py --experiment baseline_r8_lr2e4 --smoke-test
```

Depois execute os três experimentos:

```powershell
python src/train.py --experiment baseline_r8_lr2e4
python src/train.py --experiment lower_lr_r8
python src/train.py --experiment larger_rank_r16
```

Como os testes de duas épocas ainda trocaram números e prazos, foi executada
uma quarta configuração de refinamento:

```powershell
python src/train.py --experiment final_r16_10epochs
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
python src/evaluate.py --name base
python src/evaluate.py --adapter artifacts/baseline_r8_lr2e4 --name baseline_r8
python src/evaluate.py --adapter artifacts/lower_lr_r8 --name lower_lr
python src/evaluate.py --adapter artifacts/larger_rank_r16 --name rank_r16
python src/evaluate.py --adapter artifacts/final_r16_10epochs --name final_r16_10epochs
```

Os JSONs em `results/` contêm ROUGE-L, F1 por tokens, latência, respostas lado a
lado e campos para avaliação humana de correção, clareza e segurança (1 a 5).
ROUGE e F1 ajudam na comparação, mas não substituem a revisão factual.

## 5. Abrir o frontend

Escolha o adaptador com melhor avaliação:

```powershell
python app.py --adapter artifacts/final_r16_10epochs
```

O Gradio mostrará o endereço local. A interface permite controlar temperatura,
top-p e limite de tokens, e exibe latência e pico de VRAM.

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

- As regras são sintéticas e não podem ser usadas como orientação real.
- O dataset é pequeno e avalia principalmente memorização e paráfrase.
- O modelo pode alucinar; em produção seria necessário recuperar os documentos
  oficiais (RAG), citar fontes e manter revisão humana.
- A avaliação humana deve ser preenchida sem saber qual configuração produziu
  cada resposta, reduzindo viés na escolha do melhor experimento.
