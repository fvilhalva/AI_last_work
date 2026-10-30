import numpy as np

# Alunos: Felipe Echeverria, Alexandre Arruda e Sizenando

# Dados de entrada: [X1, X2, Bias]
X = np.array([
    [20.0, 10.0, 1.0],   # R1
    [80.0, 60.0, 1.0],   # R2
    [-20.0, 10.0, 1.0]   # R3
])

# Rótulos pretendidos (d)
d = np.array([0, 1, 1])

# Nomes dos registos
nomes = ["R1", "R2", "R3"]

# Hiperparâmetros e inicialização
taxa_aprendizagem = 0.2
pesos = np.array([0.1, 0.1, 0.1])  # [w1, w2, wv]
epocas_max = 5

print(f"{'Reg':<4} | {'w1':>7} {'w2':>7} {'wv':>7} | {'net':>8} | {'Y':>2} | {'d':>2} | {'erro':>4} | {'Δw1':>6} {'Δw2':>6} {'Δwv':>6}")
print("-" * 75)

for epoca in range(epocas_max):
    houve_erro = False
    
    for i in range(len(X)):
        xi = X[i]
        alvo = d[i]
        
        # Guarda os pesos antes do ajuste para exibição
        w_ant = pesos.copy()
        
        # 1. Cálculo da soma ponderada (net)
        net = np.dot(xi, pesos)
        
        # 2. Função de ativação (limiar em 0)
        y = 1 if net > 0 else 0
        
        # 3. Cálculo do erro
        erro = alvo - y
        
        # 4. Cálculo dos gradientes / ajustes (Δw)
        delta_w = taxa_aprendizagem * erro * xi
        
        # 5. Atualização dos pesos
        pesos += delta_w
        
        if erro != 0:
            houve_erro = True

        print(f"{nomes[i]:<4} | {w_ant[0]:7.2f} {w_ant[1]:7.2f} {w_ant[2]:7.2f} | {net:8.1f} | {y:2d} | {alvo:2d} | {erro:4d} | {delta_w[0]:6.1f} {delta_w[1]:6.1f} {delta_w[2]:6.1f}")
    
    # Se uma época inteira passar sem erros, o modelo convergiu
    if not houve_erro:
        print(f"\nO modelo convergiu com sucesso na época {epoca + 1}!")
        break

print("\nPesos finais ajustados:")
print(f"w1 = {pesos[0]:.4f}, w2 = {pesos[1]:.4f}, viés (wv) = {pesos[2]:.4f}")