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
