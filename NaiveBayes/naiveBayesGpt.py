from collections import defaultdict

# Dados de treino
dataset = [
    {"Outlook": "Rainy", "Temperature": "Hot", "Humidity": "High", "Windy": False, "PlayGolf": "No"},
    {"Outlook": "Rainy", "Temperature": "Hot", "Humidity": "High", "Windy": True, "PlayGolf": "No"},
    {"Outlook": "Overcast", "Temperature": "Hot", "Humidity": "High", "Windy": False, "PlayGolf": "Yes"},
    {"Outlook": "Sunny", "Temperature": "Mild", "Humidity": "High", "Windy": False, "PlayGolf": "Yes"},
    {"Outlook": "Sunny", "Temperature": "Cool", "Humidity": "Normal", "Windy": False, "PlayGolf": "Yes"},
    {"Outlook": "Sunny", "Temperature": "Cool", "Humidity": "Normal", "Windy": True, "PlayGolf": "No"},
    {"Outlook": "Overcast", "Temperature": "Cool", "Humidity": "Normal", "Windy": True, "PlayGolf": "Yes"},
    {"Outlook": "Rainy", "Temperature": "Mild", "Humidity": "High", "Windy": False, "PlayGolf": "No"},
    {"Outlook": "Rainy", "Temperature": "Cool", "Humidity": "Normal", "Windy": False, "PlayGolf": "Yes"},
    {"Outlook": "Sunny", "Temperature": "Mild", "Humidity": "Normal", "Windy": False, "PlayGolf": "Yes"},
    {"Outlook": "Rainy", "Temperature": "Mild", "Humidity": "Normal", "Windy": True, "PlayGolf": "Yes"},
    {"Outlook": "Overcast", "Temperature": "Mild", "Humidity": "High", "Windy": True, "PlayGolf": "Yes"},
    {"Outlook": "Overcast", "Temperature": "Hot", "Humidity": "Normal", "Windy": False, "PlayGolf": "Yes"},
    {"Outlook": "Sunny", "Temperature": "Mild", "Humidity": "High", "Windy": True, "PlayGolf": "No"},
]

# Separa os dados por classe (Yes ou No)
def separar_por_classe(dataset):
    separado = {"Yes": [], "No": []}
    for amostra in dataset:
        classe = amostra["PlayGolf"]
        separado[classe].append(amostra)
    return separado

# Calcula as probabilidades condicionais P(Atributo=valor | Classe)
def calcular_probabilidades_condicionais(dataset):
    separado = separar_por_classe(dataset)
    probabilidades = {"Yes": defaultdict(lambda: defaultdict(int)),
                      "No": defaultdict(lambda: defaultdict(int))}

    for classe in separado:
        total = len(separado[classe])
        for amostra in separado[classe]:
            for atributo, valor in amostra.items():
                if atributo == "PlayGolf":
                    continue
                probabilidades[classe][atributo][valor] += 1

        # Convertendo contagens em probabilidades
        for atributo in probabilidades[classe]:
            total_valores = sum(probabilidades[classe][atributo].values())
            for valor in probabilidades[classe][atributo]:
                probabilidades[classe][atributo][valor] /= total

    return probabilidades

# Probabilidades das classes P(Yes) e P(No)
def calcular_probabilidades_classes(dataset):
    total = len(dataset)
    sim = sum(1 for d in dataset if d["PlayGolf"] == "Yes")
    nao = total - sim
    return {"Yes": sim / total, "No": nao / total}

# Função para prever com base nos atributos de entrada
def prever_amostra(amostra, probabilidades_cond, prob_classes):
    resultados = {}

    for classe in ["Yes", "No"]:
        prob_total = prob_classes[classe]

        for atributo, valor in amostra.items():
            prob_atributo = probabilidades_cond[classe][atributo].get(valor, 1e-6)  # evita prob=0
            prob_total *= prob_atributo

        resultados[classe] = prob_total

    # Normaliza as probabilidades para somarem 1
    total = sum(resultados.values())
    for classe in resultados:
        resultados[classe] /= total

    return resultados

# ----- Execução -----

# 1. Treinamento
prob_cond = calcular_probabilidades_condicionais(dataset)
prob_classes = calcular_probabilidades_classes(dataset)

# 2. Previsão
nova_amostra = {"Outlook": "Rainy", "Temperature": "Hot", "Humidity": "Normal", "Windy": False}
resultado = prever_amostra(nova_amostra, prob_cond, prob_classes)

# 3. Exibição
print("Probabilidades para jogar golfe:")
for classe, prob in resultado.items():
    print(f"{classe}: {prob:.2%}")

print("\nDecisão final:", "Jogar" if resultado["Yes"] > resultado["No"] else "Não Jogar")
