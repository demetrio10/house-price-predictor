
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error


# 1. Citim datasetul
df = pd.read_csv("data/bucharest_apartments_clean_stage1.csv")

# 2. Aplicam aceleasi filtre ca in experimentele anterioare
df = df[
    (df["area_sqm"] >= 20) &
    (df["area_sqm"] <= 200) &
    (df["price_eur"] >= 20000) &
    (df["price_eur"] <= 500000)
].copy()

# Inspectam etajele fara sa modificam datele sau caracteristicile modelului.
print("\nCele mai frecvente 30 valori pentru floor:")

print(
    df["floor"]
    .value_counts(dropna=False)
    .head(30)
    .to_string()
)

print("\n20 exemple de valori lungi:")

floor_text = df["floor"].dropna().astype(str)

examples = (
    floor_text[floor_text.str.len() > 25]
    .drop_duplicates()
    .head(20)
)

for value in examples:
    print(repr(value))

# 3. Alegem caracteristicile si targetul
numeric_features = ["area_sqm", "rooms"]
categorical_features = ["area"]

features = numeric_features + categorical_features

X = df[features]
y = df["price_eur"]

# 4. Impartim datele in 60% train, 20% validation, 20% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.25,
    random_state=42
)


# Calculam limitele pretului pe mp doar din train, fara validation sau test.
train_price_per_sqm = (
    y_train / X_train["area_sqm"]
)

lower = train_price_per_sqm.quantile(0.01)
upper = train_price_per_sqm.quantile(0.99)

print(f"Limita inferioara: {lower:.2f} EUR/m2")
print(f"Limita superioara: {upper:.2f} EUR/m2")

mask = train_price_per_sqm.between(lower, upper)

# Pastram separat train-ul filtrat pentru antrenarea configuratiilor.
X_train_filtered = X_train.loc[mask]
y_train_filtered = y_train.loc[mask]

print("Train initial:", len(X_train))
print("Train filtrat:", len(X_train_filtered))


# 5. Construim un preprocesor nou pentru fiecare model
def create_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "num",
                SimpleImputer(strategy="median"),
                numeric_features
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            )
        ]
    )


# 6. Configuratii pentru Gradient Boosting
configs = [
    {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 2},
    {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 3},
    {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 3},
    {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 4}
]

results = []
best_model = None
best_val_mae = float("inf")

# 7. Antrenam si evaluam fiecare configuratie
for config in configs:
    model = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            ("regression", GradientBoostingRegressor(
                **config,
                loss="squared_error",
                min_samples_leaf=10,
                random_state=42
            ))
        ]
    )

    model.fit(X_train_filtered, y_train_filtered)

    train_pred = model.predict(X_train_filtered)
    val_pred = model.predict(X_val)

    train_mae = mean_absolute_error(
        y_train_filtered,
        train_pred
    )
    val_mae = mean_absolute_error(y_val, val_pred)

    results.append({
        "config": config,
        "train_mae": train_mae,
        "val_mae": val_mae
    })

    if val_mae < best_val_mae:
        best_val_mae = val_mae
        best_model = model

    print(
        f"Arbori={config['n_estimators']:3d} | "
        f"Learning Rate={config['learning_rate']:.2f} | "
        f"Depth={config['max_depth']} | "
        f"Train MAE={train_mae:,.2f} | "
        f"Validation MAE={val_mae:,.2f}"
    )


# 8. Selectam configuratia cu cel mai mic MAE validation
best_result = min(results, key=lambda r: r["val_mae"])
best_config = best_result["config"]

print("\nCea mai buna configuratie:", best_config)
print(f"MAE Validation: {best_val_mae:,.2f} EUR")


# 9. Reantrenam modelul selectat pe train + validation
final_model = Pipeline(
    steps=[
        ("preprocessor", create_preprocessor()),
        ("regression", GradientBoostingRegressor(
            **best_config,
            loss="squared_error",
            min_samples_leaf=10,
            random_state=42
        ))
    ]
)

final_model.fit(X_temp, y_temp)


# 10. Evaluam o singura data modelul final pe test
test_pred = final_model.predict(X_test)
test_mae = mean_absolute_error(y_test, test_pred)

print(f"MAE final pe test: {test_mae:,.2f} EUR")

# Construim un tabel cu caracteristicile apartamentelor din test.
analysis = X_test.copy()

# Adaugam pretul real si predictia.
analysis["price_real"] = y_test
analysis["price_predicted"] = test_pred

# Un reziduu pozitiv inseamna ca modelul a subestimat pretul.
analysis["residual"] = (
    analysis["price_real"] - analysis["price_predicted"]
)
analysis["absolute_error"] = analysis["residual"].abs()

# Calculam pretul real pe metru patrat.
analysis["price_per_sqm"] = (
    analysis["price_real"] / analysis["area_sqm"]
)

print("\nStatistici pret pe metru patrat:")
print(
    analysis["price_per_sqm"].describe(
        percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    )
)

print("\nTop 10 cele mai mici preturi pe metru patrat:")
print(
    analysis.sort_values("price_per_sqm")
    [["area_sqm", "rooms", "area", "price_real", "price_per_sqm"]]
    .head(10)
    .to_string(index=False)
)

print("\nTop 10 cele mai mari preturi pe metru patrat:")
print(
    analysis.sort_values("price_per_sqm", ascending=False)
    [["area_sqm", "rooms", "area", "price_real", "price_per_sqm"]]
    .head(10)
    .to_string(index=False)
)

# Salvam rezultatele separat de datasetul folosit la antrenare.
analysis.to_csv(
    "data/gradient_boosting_errors.csv",
    index=False
)

print("\nTop 10 cele mai mari erori:")
print(
    analysis.sort_values(
        "absolute_error",
        ascending=False
    ).head(10).to_string(index=False)
)


# 11. Prezicem pretul unui apartament nou
apartment = pd.DataFrame([{
    "area_sqm": 70,
    "rooms": 3,
    "area": "militari"
}])

prediction = final_model.predict(apartment)[0]

print(
    f"Pret estimat pentru 70 m², "
    f"3 camere, Militari: {prediction:,.2f} EUR"
)
