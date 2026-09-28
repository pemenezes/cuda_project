# Dataset sintético

Este dataset representa a **Faculdade Horizonte**, uma instituição fictícia.
As regras não pertencem à FIAP nem a qualquer outra instituição real e não
devem orientar decisões acadêmicas.

O arquivo `policies.json` é a fonte canônica. Cada regra recebe cinco
formulações de pergunta em `src/prepare_data.py`:

- três para treino;
- uma para validação;
- uma para teste.

Essa separação mede se o modelo consegue responder a uma formulação não vista,
mas não mede generalização para regulamentos novos. Há ainda exemplos fora de
escopo para ensinar e testar uma recusa segura.

Para um uso real, as regras sintéticas devem ser substituídas por documentos
oficiais, com autorização, versionamento, referência da fonte e revisão por uma
pessoa responsável da instituição.
