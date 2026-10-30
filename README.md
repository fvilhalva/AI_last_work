# AI_last_work

## Executar

Com Python 3 instalado, execute:

```bash
./run.sh
```

O script cria o ambiente virtual `.venv` automaticamente, instala as dependências
e executa o perceptron.

Ao iniciar, digite a quantidade de registros, os valores `X1`, `X2`, o rótulo
(`0` ou `1`), a taxa de aprendizagem e o número máximo de épocas.

Para executar automaticamente o teste com dados digitados, use:

```bash
./run.sh < teste_entrada.txt
```