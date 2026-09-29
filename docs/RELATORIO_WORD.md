# CP IA — Treinamento de IA utilizando CUDA

## Integrantes

- Anny Carolina — RM98295
- Eric Carvalho — RM550249
- Henrique Lima — RM551528
- Pedro Menezes — RM97432
- Renato Izumi — RM99242

## Repositório

https://github.com/pemenezes/cuda_project

## 1. Introdução

Este projeto apresenta o fine-tuning local de uma Large Language Model (LLM)
utilizando uma GPU NVIDIA e a plataforma CUDA. O objetivo foi criar o
**Barista de Apartamento**, um assistente especializado em orientar iniciantes
no preparo de cafés e bebidas simples em casa.

Em vez de treinar uma LLM do zero, processo que exigiria grande quantidade de
dados, milhares de GPUs e alto custo, foi utilizado um modelo já pré-treinado.
Esse modelo recebeu um treinamento adicional com um dataset pequeno e
especializado, permitindo adaptar seu comportamento ao domínio escolhido.

O projeto inclui criação do dataset, fine-tuning com QLoRA, comparação de
hiperparâmetros, avaliação automática e qualitativa, interface web local e
comprovação de que os testes foram executados na GPU.

## 2. Problema escolhido

O problema definido foi responder dúvidas práticas sobre preparo doméstico de
café. O assistente recebeu conhecimentos sobre:

- café coado;
- prensa francesa;
- cafeteira italiana;
- café solúvel e cold brew;
- cappuccino, latte, mocha e café gelado;
- proporções entre café e água;
- moagem e temperatura;
- correção de café amargo, ácido, forte ou fraco;
- armazenamento e conservação;
- leite vegetal e bebidas sem cafeína;
- perguntas fora do escopo.

O tema foi mantido propositalmente restrito. Isso possibilita criar respostas
esperadas, identificar erros e comparar objetivamente o modelo antes e depois
do fine-tuning.

## 3. Modelo selecionado

Foi escolhido o modelo `Qwen/Qwen2.5-0.5B-Instruct`, disponibilizado no
Hugging Face.

O modelo possui aproximadamente 0,49 bilhão de parâmetros e já foi preparado
para seguir instruções e participar de conversas. A escolha foi motivada pelos
seguintes fatores:

- suporte a português;
- formato adequado para chat;
- compatibilidade com Transformers, PEFT e LoRA;
- tamanho apropriado para uma GPU com 6 GB de VRAM;
- possibilidade de executar vários experimentos em tempo reduzido;
- licença e arquivos disponíveis para execução local.

Modelos maiores poderiam apresentar melhor capacidade de generalização, mas
consumiriam mais memória e aumentariam o tempo de treinamento. O Qwen 0.5B foi
uma escolha compatível com o hardware disponível e com o objetivo didático.

## 4. Hardware e ambiente CUDA

O projeto foi executado com a seguinte configuração:

- GPU: NVIDIA GeForce RTX 2060;
- VRAM dedicada: 6 GB;
- compute capability: 7.5;
- arquitetura da GPU: Turing;
- driver NVIDIA: 616.64;
- compatibilidade máxima informada pelo driver: CUDA 13.4;
- CUDA Toolkit instalado: 13.0, versão V13.0.48;
- Python: 3.12.2;
- PyTorch: 2.14.0+cu130;
- runtime CUDA utilizado pelo PyTorch: 13.0.

O CUDA 13.4 exibido pelo `nvidia-smi` representa a maior versão suportada pelo
driver. O Toolkit e o runtime do PyTorch utilizam CUDA 13.0. Essa combinação é
compatível porque o driver instalado suporta aplicações construídas com
versões anteriores do runtime CUDA.

O comando `nvcc --version` confirmou a instalação do compilador do CUDA
Toolkit. O script `scripts/check_gpu.py` realizou uma multiplicação matricial
FP16 diretamente na GPU e confirmou que o resultado estava no dispositivo
`cuda:0`.

### Evidência da GPU

[INSERIR FIGURA 1 — Terminal integrado do VSCode executando `nvidia-smi`]

**Figura 1 —** Detecção da RTX 2060, driver 616.64, memória utilizada e processo
Python em execução.

[INSERIR FIGURA 2 — Terminal executando `nvcc --version`]

**Figura 2 —** CUDA Toolkit 13.0 e compilador NVCC V13.0.48.

[INSERIR FIGURA 3 — Terminal executando `python scripts/check_gpu.py`]

**Figura 3 —** PyTorch reconhecendo CUDA, compute capability 7.5, 6 GB de VRAM
e cálculo matricial executado em `cuda:0`.

## 5. Dataset

O dataset foi criado especificamente para o projeto. A fonte canônica está no
arquivo `data/barista_knowledge.json`.

Sua versão final contém:

- 30 conhecimentos canônicos;
- 134 exemplos de treinamento;
- 33 exemplos de validação;
- 33 exemplos de teste;
- paráfrases das perguntas;
- perguntas com linguagem natural;
- perguntas fora do escopo;
- diálogos com histórico e perguntas consecutivas.

Os splits são gerados por `src/prepare_data.py` com seed fixa, permitindo
reproduzir a divisão. As respostas do conjunto de teste não são utilizadas
diretamente durante o treinamento.

O dataset é demonstrativo. Ele não substitui orientações médicas,
nutricionais ou profissionais de segurança alimentar.

## 6. Fine-tuning com QLoRA

Fine-tuning é o processo de continuar o treinamento de um modelo existente com
dados especializados. Neste projeto foi utilizada a técnica QLoRA.

O modelo principal foi carregado com quantização de 4 bits e seus pesos foram
mantidos congelados. Pequenas matrizes LoRA foram inseridas nas camadas de
atenção e projeção. Somente essas matrizes foram atualizadas pelo treinamento.

Essa abordagem reduz:

- uso de VRAM;
- tempo de treinamento;
- quantidade de parâmetros treináveis;
- tamanho do arquivo resultante.

O treinamento também utilizou FP16, gradient checkpointing, otimizador de 8
bits e gradient accumulation. O código impede a execução do treinamento caso
o CUDA não esteja disponível.

## 7. Experimentos

Foram executadas quatro configurações:

1. Rank LoRA 8, learning rate 0,0002 e 2 épocas.
2. Rank LoRA 8, learning rate 0,0001 e 2 épocas.
3. Rank LoRA 16, learning rate 0,0002 e 2 épocas.
4. Rank LoRA 16, learning rate 0,0002 e 8 épocas.

Os três primeiros experimentos permitem comparar learning rate e capacidade
do adaptador. A quarta configuração foi criada depois que os treinamentos de
duas épocas ainda apresentaram respostas imprecisas.

A melhor configuração foi rank 16, learning rate 0,0002 e 8 épocas. Seu
treinamento levou aproximadamente 648,8 segundos, ou 10,8 minutos, e utilizou
pico de aproximadamente 1,35 GiB de VRAM.

## 8. Avaliação

O modelo base e os adaptadores foram avaliados com o mesmo conjunto de teste e
geração determinística.

Foram utilizadas:

- loss de validação;
- ROUGE-L;
- F1 por tokens;
- latência de geração;
- inspeção humana das respostas;
- comparação entre resposta esperada e resposta produzida.

Resultados obtidos:

- Modelo base: ROUGE-L 0,126 e F1 0,159.
- Rank 8, learning rate 0,0002: ROUGE-L 0,222 e F1 0,298.
- Rank 8, learning rate 0,0001: ROUGE-L 0,172 e F1 0,217.
- Rank 16, 2 épocas: ROUGE-L 0,255 e F1 0,314.
- Rank 16, 8 épocas: ROUGE-L 0,927 e F1 0,930.

Os resultados mostram uma melhora significativa após o fine-tuning. Entretanto,
as métricas automáticas não garantem que todas as respostas sejam corretas ou
seguras. Por isso também foi realizada uma análise qualitativa.

## 9. Comparação antes e depois

O script `compare.py` desativa temporariamente o adaptador LoRA para obter a
resposta do modelo original e depois ativa o adaptador para gerar a resposta
ajustada. Os dois modelos recebem o mesmo prompt de sistema, a mesma pergunta e
parâmetros determinísticos.

### 9.1 Cappuccino sem máquina

**Pergunta:** Como faço cappuccino sem máquina?

O modelo base inventou medidas, utilizou termos incorretos e produziu uma
resposta incompleta. O modelo ajustado respondeu:

> Combine uma parte de café forte, uma parte de leite quente e uma parte de
> espuma de leite. Canela ou cacau são opcionais.

Esse caso demonstra melhora de correção, objetividade e clareza.

### 9.2 Café amargo

**Pergunta:** Por que meu café ficou amargo?

O modelo base sugeriu informações incorretas sobre água fria, açúcar e amônia.
O modelo ajustado respondeu:

> Se o café ficou muito amargo, tente moagem mais grossa, água menos quente ou
> tempo de contato menor.

A resposta ajustada está alinhada ao conhecimento canônico do dataset.

### 9.3 Pergunta fora do escopo

**Pergunta:** Ensine uma receita de coquetel alcoólico.

O modelo base inventou uma receita inadequada. O modelo com fine-tuning também
falhou, pois deveria recusar a solicitação, mas criou uma preparação com café e
leite.

Esse resultado foi mantido na documentação por representar uma limitação real.
Ele demonstra que o fine-tuning melhorou o domínio principal, mas os exemplos
de recusa ainda são insuficientes para garantir comportamento seguro em todas
as perguntas fora do escopo.

[INSERIR FIGURA 4 — Comparação do cappuccino no terminal]

[INSERIR FIGURA 5 — Comparação do café amargo no terminal]

[INSERIR FIGURA 6 — Falha na pergunta sobre coquetel]

## 10. Frontend

Foi desenvolvido um frontend local utilizando Gradio. A aplicação permite:

- escrever perguntas;
- manter histórico entre mensagens;
- fazer perguntas consecutivas;
- ajustar temperatura;
- ajustar top-p;
- definir o máximo de tokens;
- visualizar tempo de resposta;
- visualizar pico de VRAM.

Durante o desenvolvimento foi corrigido um problema no qual o Gradio
armazenava o histórico como blocos de conteúdo, mas o template do Qwen esperava
texto. Após a normalização desse histórico, o frontend passou a aceitar várias
perguntas consecutivas.

[INSERIR FIGURA 7 — Tela principal do Barista de Apartamento]

[INSERIR FIGURA 8 — Conversa com perguntas consecutivas]

## 11. Limitações

As principais limitações identificadas foram:

- dataset pequeno;
- forte aprendizado das formulações presentes no dataset;
- dificuldade maior em diálogos reservados;
- falha de recusa em uma pergunta fora do escopo;
- possibilidade de alucinação;
- modelo base pequeno, com 0,5 bilhão de parâmetros;
- ausência de consulta a fontes externas atualizadas.

Uma versão de produção deveria utilizar um dataset maior, mais exemplos
multi-turno, mais exemplos negativos, revisão especializada e RAG para
consultar fontes confiáveis.

## 12. Conclusão

O projeto demonstrou que é possível adaptar localmente uma LLM utilizando uma
GPU doméstica. A RTX 2060 executou preparação, fine-tuning, avaliação e
inferência sem falta de memória.

O fine-tuning QLoRA produziu melhora clara nas perguntas pertencentes ao
domínio. O F1 passou de 0,159 no modelo base para 0,930 no melhor adaptador, e
as respostas sobre cappuccino e correção de amargor se tornaram objetivas e
coerentes.

Ao mesmo tempo, a falha na pergunta sobre coquetel mostrou que uma média alta
não elimina erros individuais. Portanto, avaliação humana e análise de
limitações continuam essenciais.

Todo o código, dataset, configurações, testes e resultados leves está
documentado no GitHub para permitir reprodução.

## 13. Execução

Para abrir o frontend:

```powershell
.\.venv\Scripts\Activate.ps1
python app.py
```

A aplicação fica disponível em:

```text
http://127.0.0.1:7860
```

Para comparar o modelo base com o modelo ajustado:

```powershell
python compare.py --question "Como faço cappuccino sem máquina?"
```
