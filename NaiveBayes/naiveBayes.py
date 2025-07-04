from collections import defaultdict
import matplotlib.pyplot as plt

# --- 1. Dataset ---
dataset = [
    {"Outlook": "Rainy", "Temperature": "Hot", "Humidity": "High", "Windy": "Weak", "Play": "No"},
    {"Outlook": "Rainy", "Temperature": "Hot", "Humidity": "High", "Windy": "Strong", "Play": "No"},
    {"Outlook": "Overcast", "Temperature": "Hot", "Humidity": "High", "Windy": "Weak", "Play": "Yes"},
    {"Outlook": "Sunny", "Temperature": "Mild", "Humidity": "High", "Windy": "Weak", "Play": "Yes"},
    {"Outlook": "Sunny", "Temperature": "Cool", "Humidity": "Normal", "Windy": "Weak", "Play": "Yes"},
    {"Outlook": "Sunny", "Temperature": "Cool", "Humidity": "Normal", "Windy": "Strong", "Play": "No"},
    {"Outlook": "Overcast", "Temperature": "Cool", "Humidity": "Normal", "Windy": "Strong", "Play": "Yes"},
    {"Outlook": "Rainy", "Temperature": "Mild", "Humidity": "High", "Windy": "Weak", "Play": "No"},
    {"Outlook": "Rainy", "Temperature": "Cool", "Humidity": "Normal", "Windy": "Weak", "Play": "Yes"},
    {"Outlook": "Sunny", "Temperature": "Mild", "Humidity": "Normal", "Windy": "Weak", "Play": "Yes"},
    {"Outlook": "Rainy", "Temperature": "Mild", "Humidity": "Normal", "Windy": "Strong", "Play": "Yes"},
    {"Outlook": "Overcast", "Temperature": "Mild", "Humidity": "High", "Windy": "Strong", "Play": "Yes"},
    {"Outlook": "Overcast", "Temperature": "Hot", "Humidity": "Normal", "Windy": "Weak", "Play": "Yes"},
    {"Outlook": "Sunny", "Temperature": "Mild", "Humidity": "High", "Windy": "Strong", "Play": "No"},
]

# --- 2. Pré-processamento ---

def separar_por_classe(dataset):
    separado = {"Yes": [], "No": []}
    for amostra in dataset:
        classe = amostra["Play"]
        separado[classe].append(amostra)
    return separado

def contar_valores_possiveis(dataset):
    valores = defaultdict(set)
    for d in dataset:
        for feature in d:
            if feature == "Play":
                continue
            valores[feature].add(d[feature])
    return {k: list(v) for k, v in valores.items()}

def contar_condicional(dataset):
    separado = separar_por_classe(dataset)
    contagem = {"Yes": defaultdict(lambda: defaultdict(int)),
                "No": defaultdict(lambda: defaultdict(int))}

    for classe in separado:
        for amostra in separado[classe]:
            for feature, valor in amostra.items():
                if feature != "Play":
                    contagem[classe][feature][valor] += 1
    return contagem

def calcular_prob_classes(dataset):
    total = len(dataset)
    yes = sum(1 for d in dataset if d["Play"] == "Yes")
    no = total - yes
    return {"Yes": yes / total, "No": no / total}, {"Yes": yes, "No": no}

# --- 3. Classificação ---

def classificar(amostra, contagens, prob_classes, total_por_classe, valores_possiveis):
    resultados = {}

    for classe in ["Yes", "No"]:
        prob_total = prob_classes[classe]

        for feature, valor in amostra.items():
            count = contagens[classe][feature].get(valor, 0) + 1  # Laplace
            total = total_por_classe[classe] + len(valores_possiveis[feature])  # Laplace denom
            prob_total *= count / total

        resultados[classe] = prob_total

    # Normaliza
    total_prob = resultados["Yes"] + resultados["No"]
    resultados["Yes"] /= total_prob
    resultados["No"] /= total_prob

    return resultados

# --- 4. Visualização ---

def plotar_resultados(testes, contagens, prob_classes, total_por_classe, valores_possiveis):
    n = len(testes)
    cols = 2  # Número de colunas (ajustável)
    rows = (n + cols - 1) // cols  # Número de linhas necessárias

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()  # Para iterar facilmente mesmo se for 1 linha só

    for i, caso in enumerate(testes):
        resultado = classificar(caso, contagens, prob_classes, total_por_classe, valores_possiveis)

        labels = ["Yes", "No"]
        values = [resultado["Yes"], resultado["No"]]

        bars = axes[i].bar(labels, values, color=["green", "red"])
        axes[i].set_ylim(0, 1)
        titulo = f"Entrada {i+1}:\n{caso['Outlook']}, {caso['Temperature']}, {caso['Humidity']}, {caso['Windy']}"
        axes[i].set_title(titulo, fontsize=10)
        axes[i].set_ylabel("Probabilidade")
        axes[i].bar_label(bars, fmt="%.2f", label_type='edge')

    # Esconde eixos extras (se houver)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


# --- 5. Execução principal ---

valores_possiveis = contar_valores_possiveis(dataset)
contagens = contar_condicional(dataset)
prob_classes, total_por_classe = calcular_prob_classes(dataset)

# Entradas de teste
testes = [
    {"Outlook": "Rainy", "Temperature": "Hot", "Humidity": "Normal", "Windy": "Weak"},
    {"Outlook": "Sunny", "Temperature": "Cool", "Humidity": "High", "Windy": "Strong"},
    {"Outlook": "Overcast", "Temperature": "Hot", "Humidity": "Normal", "Windy": "Weak"},
    {"Outlook": "Rainy", "Temperature": "Mild", "Humidity": "High", "Windy": "Strong"}
]

# Mostrar resultados em gráfico
plotar_resultados(testes, contagens, prob_classes, total_por_classe, valores_possiveis)
