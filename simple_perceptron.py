import numpy as np

# Alunos: Felipe Echeverria, Alexandre Arruda e Sizenando


def treinar_perceptron(
    X: np.ndarray,
    d: np.ndarray,
    nomes: list[str],
    taxa_aprendizagem: float,
    epocas_max: int,
) -> None:
    pesos = np.array([0.1, 0.1, 0.1], dtype=float)

    print(
        f"{'Reg':<4} | {'w1':>7} {'w2':>7} {'wv':>7} | {'net':>8} | "
        f"{'Y':>2} | {'d':>2} | {'erro':>4} | {'Δw1':>6} {'Δw2':>6} {'Δwv':>6}"
    )
    print("-" * 75)

    for epoca in range(epocas_max):
        houve_erro = False

        for i in range(len(X)):
            xi = X[i]
            alvo = int(d[i])
            w_ant = pesos.copy()
            net = np.dot(xi, pesos)
            y = 1 if net > 0 else 0
            erro = alvo - y
            delta_w = taxa_aprendizagem * erro * xi
            pesos += delta_w

            if erro != 0:
                houve_erro = True

            print(
                f"{nomes[i]:<4} | {w_ant[0]:7.2f} {w_ant[1]:7.2f} "
                f"{w_ant[2]:7.2f} | {net:8.1f} | {y:2d} | {alvo:2d} | "
                f"{erro:4d} | {delta_w[0]:6.1f} {delta_w[1]:6.1f} "
                f"{delta_w[2]:6.1f}"
            )

        if not houve_erro:
            print(f"\nO modelo convergiu com sucesso na época {epoca + 1}!")
            break
    else:
        print(f"\nO modelo não convergiu nas {epocas_max} épocas.")

    print("\nPesos finais ajustados:")
    print(f"w1 = {pesos[0]:.4f}, w2 = {pesos[1]:.4f}, viés (wv) = {pesos[2]:.4f}")


def ler_float(mensagem: str) -> float:
    while True:
        try:
            return float(input(mensagem).replace(",", "."))
        except ValueError:
            print("Digite um número válido.")


def ler_inteiro(mensagem: str, minimo: int = 1) -> int:
    while True:
        try:
            valor = int(input(mensagem))
            if valor < minimo:
                print(f"Digite um inteiro maior ou igual a {minimo}.")
                continue
            return valor
        except ValueError:
            print("Digite um inteiro válido.")


def ler_rotulo(mensagem: str) -> int:
    while True:
        valor = input(mensagem).strip()
        if valor in {"0", "1"}:
            return int(valor)
        print("O rótulo deve ser 0 ou 1.")


def dados_digitados() -> tuple[np.ndarray, np.ndarray, list[str]]:
    quantidade = ler_inteiro("\nQuantos registros deseja cadastrar? ")
    registros = []
    rotulos = []

    for i in range(quantidade):
        print(f"\nRegistro R{i + 1}")
        x1 = ler_float("Digite X1: ")
        x2 = ler_float("Digite X2: ")
        rotulo = ler_rotulo("Digite o rótulo desejado (0 ou 1): ")
        registros.append([x1, x2, 1.0])
        rotulos.append(rotulo)

    return np.array(registros), np.array(rotulos), [
        f"R{i + 1}" for i in range(quantidade)
    ]


def main() -> None:
    print("=== Perceptron simples ===")
    X, d, nomes = dados_digitados()
    taxa_aprendizagem = ler_float("\nTaxa de aprendizagem (ex.: 0.2): ")
    while taxa_aprendizagem <= 0:
        print("A taxa de aprendizagem deve ser maior que zero.")
        taxa_aprendizagem = ler_float("Taxa de aprendizagem: ")
    epocas_max = ler_inteiro("Número máximo de épocas (ex.: 10): ")

    print()
    treinar_perceptron(X, d, nomes, taxa_aprendizagem, epocas_max)


if __name__ == "__main__":
    main()
