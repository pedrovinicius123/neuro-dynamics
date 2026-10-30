# Neuro Dynamics

Projeto C de redes neurais spiking com leitura de arquiteturas, serializacao de modelos e captura de audio opcional.

## Build e testes

```sh
make compile
make test
```

## Uso

Carregar uma arquitetura e salvar um modelo:

```sh
./neura --crfile neura_architectures/n1.neur --sfile /tmp/n1.model
```

Carregar um modelo existente:

```sh
./neura --rfile /tmp/n1.model
```

A captura de audio e opt-in e permanece desligada no fluxo acima. Para tentar usar o dispositivo padrao, acrescente `--audio`.

## Treino com texto e datasets científicos

Compile o backend C e instale as dependências Python com `make compile` e `pip install -r requirements.txt`. O Python busca/tokeniza os dados e envia um token UTF-8 por linha ao executável `neura`; codificação em spikes, propagação LIF, atualização STDP e gravação do modelo acontecem no C. Sem uma fonte posicional, o pipeline transmite artigos do corpus científico `ccdv/arxiv-summarization` da Hugging Face (Parquet, split `train`) e treina todos os arquivos `.model` encontrados em `models/`:

```sh
python3 token_pipeline.py --output benchmark.json
```

Também é possível usar texto local, stdin ou HTTP(S), treinando um modelo específico existente ou criando/carregando o modelo correspondente a uma arquitetura `.neur`:

```sh
python3 token_pipeline.py dados.txt --model models/n1.model --output benchmark.json
python3 token_pipeline.py - --architecture neura_architectures/n1.neur --output benchmark.json
python3 token_pipeline.py --dataset ccdv/pubmed-summarization --split train --text-column article --max-examples 100
python3 token_pipeline.py dados.txt --model models/n1.model --log logs/text-training.neur
```

O encoder C transforma n-gramas de cada token em spikes da camada de entrada. O `.model` treinado é gravado no caminho de entrada ou, para uma arquitetura, em `models/<nome>.model`. Use `--model-output` para escolher outro destino. O log binário detalhado é opcional via `--log`; métricas resumidas são incluídas no benchmark. `--max-examples`, `--max-chars` e `--max-tokens` limitam o consumo de dados e processamento.

O JSON `neuro-dynamics-benchmark-v1` contém métricas, atividade por camada e candidatos de próximo token. Para analisar spikes gravados pelo executável, use:

```sh
python3 parser.py logs/log.1.neur --analyze --json spikes.json
```

## Chat de diagnóstico

Após treinar, o pipeline grava `<modelo>.chat.json` com as transições aprendidas. Inicie uma sessão com:

```sh
python3 token_pipeline.py --chat --model models/n3.model
```

Use `/reset` para limpar o contexto e `/sair` para terminar. O CLI executa cada prompt no C em modo de inferência sem STDP, mostra a atividade de spikes e permite registrar avaliação correta/incorreta em `chat_feedback.jsonl`. A resposta textual é uma continuação baseada nas transições do corpus, não uma decodificação dos pesos STDP; o benchmark antigo pode ser usado com `--output benchmark.json`, mas contém somente candidatos limitados.
