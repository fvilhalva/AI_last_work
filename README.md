# AI_last_work

## Executar

Com Python 3 instalado, execute:

```bash
./run.sh
```

O script cria o ambiente virtual `.venv` automaticamente, instala as dependências
e executa o perceptron.

Ao iniciar, escolha uma das opções:

1. Executar o teste pronto, usando os dados que já estavam no código.
2. Digitar a quantidade de registros, os valores `X1`, `X2`, o rótulo (`0` ou
   `1`), a taxa de aprendizagem e o número máximo de épocas.

As duas opções exibem a tabela de treinamento e os pesos finais.

Para executar automaticamente o teste com dados digitados, use:

```bash
./run.sh < teste_entrada.txt
```