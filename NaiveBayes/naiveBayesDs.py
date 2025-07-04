from collections import defaultdict

# Dados de treinamento
data = [
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

# 1. Calcular probabilidades a priori
total = len(data)
count_yes = sum(1 for row in data if row["Play"] == "Yes")
count_no = total - count_yes

p_yes = count_yes / total
p_no = count_no / total

# 2. Calcular contagens de features para cada classe
feature_counts_yes = defaultdict(lambda: defaultdict(int))
feature_counts_no = defaultdict(lambda: defaultdict(int))

for row in data:
    play = row["Play"]
    for feature in ["Outlook", "Temperature", "Humidity", "Windy"]:
        value = row[feature]
        if play == "Yes":
            feature_counts_yes[feature][value] += 1
        else:
            feature_counts_no[feature][value] += 1

# 3. Função para calcular probabilidades condicionais com suavização
def get_probability(feature, value, play):
    if play == "Yes":
        numerator = feature_counts_yes[feature].get(value, 0) + 1  # Suavização (add-1)
        denominator = count_yes + len(feature_counts_yes[feature])
    else:
        numerator = feature_counts_no[feature].get(value, 0) + 1
        denominator = count_no + len(feature_counts_no[feature])
    return numerator / denominator

# 4. Função para classificar nova instância
def classify(outlook, temperature, humidity, windy):
    # Probabilidade para "Yes"
    p_yes_features = p_yes
    p_yes_features *= get_probability("Outlook", outlook, "Yes")
    p_yes_features *= get_probability("Temperature", temperature, "Yes")
    p_yes_features *= get_probability("Humidity", humidity, "Yes")
    p_yes_features *= get_probability("Windy", windy, "Yes")

    # Probabilidade para "No"
    p_no_features = p_no
    p_no_features *= get_probability("Outlook", outlook, "No")
    p_no_features *= get_probability("Temperature", temperature, "No")
    p_no_features *= get_probability("Humidity", humidity, "No")
    p_no_features *= get_probability("Windy", windy, "No")

    # Normalização
    total_prob = p_yes_features + p_no_features
    p_yes_normalized = p_yes_features / total_prob
    p_no_normalized = p_no_features / total_prob

    return {
        "Yes": p_yes_normalized,
        "No": p_no_normalized,
        "Decision": "Yes" if p_yes_normalized > p_no_normalized else "No"
    }

# 5. Testar o classificador
test_case = ("Rainy", "Hot", "Normal", "Weak")
result = classify(*test_case)

print("\nResultado da Classificação:")
print(f"Condições: Outlook={test_case[0]}, Temperature={test_case[1]}, Humidity={test_case[2]}, Windy={test_case[3]}")
print(f"Probabilidade de Jogar Golfe (Yes): {result['Yes']:.2%}")
print(f"Probabilidade de Não Jogar Golfe (No): {result['No']:.2%}")
print(f"Decisão: {result['Decision']}")

# 6. Testar outros casos
print("\nOutros Testes:")
test_cases = [
    ("Sunny", "Cool", "High", "Strong"),
    ("Overcast", "Hot", "Normal", "Weak"),
    ("Rainy", "Mild", "High", "Strong")
]

for case in test_cases:
    res = classify(*case)
    print(f"\nPara {case}:")
    print(f"  Yes: {res['Yes']:.2%}, No: {res['No']:.2%} → Decisão: {res['Decision']}")