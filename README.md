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

## Dados reais, tokenizacao e benchmark

O pipeline Python aceita texto local, stdin ou uma URL HTTP(S). Os tokens passam pela arquitetura `.neur` e o resultado inclui atividade por camada, spikes, taxa de disparo e candidatos de proximo token em JSON:

```sh
python3 token_pipeline.py dados.txt --architecture neura_architectures/n1.neur --output benchmark.json
python3 token_pipeline.py https://example.com --architecture neura_architectures/n1.neur --output benchmark.json
cat dados.txt | python3 token_pipeline.py - --architecture neura_architectures/n1.neur
```

O formato `neuro-dynamics-benchmark-v1` foi feito para ser consumido por modelos externos sem depender de logs de terminal. Para analisar spikes gravados pelo executavel, use:

```sh
python3 parser.py logs/log.1.neur --analyze --json spikes.json
```
