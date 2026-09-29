# Dataset Barista de Apartamento

Este é um dataset demonstrativo de receitas, métodos e orientações simples para
preparar café em casa. Ele não substitui recomendações médicas, nutricionais ou
de segurança alimentar.

O arquivo `barista_knowledge.json` contém 30 conhecimentos canônicos. Cada item
recebe cinco formulações de pergunta em `src/prepare_data.py`:

- três para treino;
- uma para validação;
- uma para teste.

Há também perguntas fora do escopo e 12 conversas com histórico. O treino usa
oito diálogos, a validação usa dois e o teste reserva dois diálogos diferentes.

Essa separação mede paráfrase e retenção de contexto, mas não demonstra
generalização completa para receitas novas. Para uso real, o conteúdo deve ser
revisado por profissionais e apoiado por fontes confiáveis.
