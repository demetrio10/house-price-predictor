
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error

# Citim datasetul curatat
df = pd.read_csv("data/bucharest_apartments_clean_stage1.csv")

# Pastram aceleasi limite pentru comparatie
df = df[
    (df["area_sqm"] >= 20) &
    (df["area_sqm"] <= 200) &
    (df["price_eur"] >= 20000) &
    (df["price_eur"] <= 500000)
].copy()

# Transformam etajul din text in numar pentru imputare si regresie.
# "Etaj 1 / 10" devine 1 (etajul apartamentului), nu 10 (inaltimea blocului).
floor_text = df["floor"].astype("string").str.lower().str.strip()
floor_text = floor_text.str.replace(r"^parter\b", "0", regex=True)
floor_number = floor_text.str.extract(
    r"^(?:etaj\s*)?(-?\d+)(?=\s*(?:/|$))",
    expand=False
)

# Textele nerecunoscute devin NaN; imputer-ul le va completa din train.
df["floor"] = pd.to_numeric(floor_number, errors="coerce").astype("float64")
print("Etaje nerecunoscute:", (floor_text.notna() & df["floor"].isna()).sum())

# Alegem 3 caracteristici
features = ["area_sqm", "rooms", "floor"]

X = df[features]
y = df["price_eur"]

# Impartim train / test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Completam valorile lipsa folosind mediana din train
imputer = SimpleImputer(strategy="median")

X_train = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=features
)

X_test = pd.DataFrame(
    imputer.transform(X_test),
    columns=features
)

# Antrenam regresia liniara multipla
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluam
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

print(f"MAE regresie multipla: {mae:,.2f} EUR")

# Vedem coeficientul fiecarei caracteristici
for feature, coefficient in zip(features, model.coef_):
    print(f"{feature}: {coefficient:,.2f}")

print(f"Intercept: {model.intercept_:,.2f}")

# Testam un apartament ipotetic
apartment = pd.DataFrame(
    [[70, 3, 4]],
    columns=features
)

print(
    "Pret estimat:",
    model.predict(apartment)[0]
)
