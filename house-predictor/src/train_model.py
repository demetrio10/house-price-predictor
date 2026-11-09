
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.dummy import DummyRegressor

# 1. Citim datele
df = pd.read_csv("data/bucharest_apartments_clean_stage1.csv")

# 2. Selectam apartamentele pentru primul experiment
df = df[
    (df["area_sqm"] >= 20) &
    (df["area_sqm"] <= 200) &
    (df["price_eur"] >= 20000) &
    (df["price_eur"] <= 500000)
].copy()

# 3. Alegem caracteristica de intrare si valoarea de prezis
X = df[["area_sqm"]]
y = df["price_eur"]

# 4. Impartim datele in train si test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 5. Construim si antrenam modelul
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Afisam coeficientii invatati
print("Panta a:", model.coef_[0])
print("Interceptul b:", model.intercept_)

# 7. Facem predictii pe datele de test
predictions = model.predict(X_test)

# 8. Calculam eroarea medie absoluta
mae = mean_absolute_error(y_test, predictions)

# Baseline-ul prezice aceeasi valoare pentru toate apartamentele:
# mediana preturilor din setul de antrenare, fara sa foloseasca suprafata.
baseline = DummyRegressor(strategy="median")
baseline.fit(X_train, y_train)

baseline_predictions = baseline.predict(X_test)
baseline_mae = mean_absolute_error(
    y_test,
    baseline_predictions
)

# Comparam modelele pe acelasi set de test. Un MAE mai mic este mai bun.
print(f"MAE regresie liniara: {mae:,.2f} EUR")
print(f"MAE baseline median: {baseline_mae:,.2f} EUR")

# 9. Prezicem pretul unui apartament de 70 m²
new_apartment = pd.DataFrame({"area_sqm": [70]})
estimated_price = model.predict(new_apartment)
print("Pret estimat pentru 70 m²:", estimated_price[0])

# 10. Afisam preturile reale din setul de test si dreapta modelului.
plt.figure(figsize=(12, 6))

sns.scatterplot(
    x=X_test["area_sqm"],
    y=y_test,
    alpha=0.15,
    s=10,
    label="Preturi reale"
)

# Construim punctele de pe dreapta modelului: pret = a * suprafata + b.
x_line = np.linspace(20, 200, 100)
y_line = model.coef_[0] * x_line + model.intercept_

plt.plot(
    x_line,
    y_line,
    color="red",
    linewidth=3,
    label="Regresie liniara"
)

plt.xlabel("Suprafata (m²)")
plt.ylabel("Pret (EUR)")
plt.title("Primul model House Predictor")
plt.legend()
plt.grid(alpha=0.2)

plt.show()
