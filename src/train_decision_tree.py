
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

# 1. Citim datele
df = pd.read_csv("data/bucharest_apartments_clean_stage1.csv")

# 2. Aplicam filtrele folosite in experimentele anterioare
df = df[
    (df["area_sqm"] >= 20) &
    (df["area_sqm"] <= 200) &
    (df["price_eur"] >= 20000) &
    (df["price_eur"] <= 500000)
].copy()

# 3. Alegem caracteristicile si targetul
numeric_features = ["area_sqm", "rooms"]
categorical_features = ["area"]

features = numeric_features + categorical_features

X = df[features]
y = df["price_eur"]

# 4. Separare: 60% train, 20% validation, 20% test
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

# 5. Preprocesare
preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# 6. Testam mai multe adancimi
depths = [2, 5, 10, 15, 20, 25, 30, 40]
results = []

for depth in depths:
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regression", DecisionTreeRegressor(
                max_depth=depth,
                min_samples_leaf=10,
                random_state=42
            ))
        ]
    )

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    train_mae = mean_absolute_error(y_train, train_pred)
    val_mae = mean_absolute_error(y_val, val_pred)

    results.append({
        "depth": depth,
        "train_mae": train_mae,
        "val_mae": val_mae
    })

    print(
        f"Depth={depth:2d} | "
        f"Train MAE={train_mae:,.2f} | "
        f"Validation MAE={val_mae:,.2f}"
    )

# 7. Alegem adancimea cu cel mai mic MAE pe validation
best_result = min(results, key=lambda result: result["val_mae"])
best_depth = best_result["depth"]

print(f"\nCea mai buna adancime: {best_depth}")

# 8. Reantrenam pe train + validation
final_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regression", DecisionTreeRegressor(
            max_depth=best_depth,
            min_samples_leaf=10,
            random_state=42
        ))
    ]
)

final_model.fit(X_temp, y_temp)

# 9. Evaluam o singura data pe test
test_pred = final_model.predict(X_test)
test_mae = mean_absolute_error(y_test, test_pred)

print(f"MAE final pe test: {test_mae:,.2f} EUR")

# 10. Estimam pretul unui apartament nou
apartment = pd.DataFrame([{
    "area_sqm": 70,
    "rooms": 3,
    "area": "militari"
}])

prediction = final_model.predict(apartment)[0]

print(f"Pret estimat pentru 70 m² in Militari: {prediction:,.2f} EUR")
