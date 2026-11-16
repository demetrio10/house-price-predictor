
import pandas as pd
from floor_features import extract_floor_features

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.inspection import permutation_importance

# 1. Citim datele
df = pd.read_csv("data/bucharest_apartments_clean_stage1.csv")
df = extract_floor_features(df)

# 2. Aplicam aceleasi filtre
df = df[
    (df["area_sqm"] >= 20) &
    (df["area_sqm"] <= 200) &
    (df["price_eur"] >= 20000) &
    (df["price_eur"] <= 500000)
].copy()

# 3. Selectam caracteristicile
numeric_features = [
    "area_sqm",
    "rooms",
    "floor_number",
    "total_floors",
    "relative_floor",
    "is_ground_floor",
    "is_top_floor"
]
categorical_features = ["area"]

features = numeric_features + categorical_features

X = df[features]
y = df["price_eur"]

# 4. Impartim datele in train / validation / test
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

# 6. Experimentam cu configuratii diferite
configs = [
    {"n_estimators": 50, "max_depth": 10, "max_features": 1.0},
    {"n_estimators": 100, "max_depth": 20, "max_features": 1.0},
    {"n_estimators": 100, "max_depth": 30, "max_features": 0.7},
    {"n_estimators": 200, "max_depth": 30, "max_features": 0.7}
]

results = []

for config in configs:
    model = Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("regression", RandomForestRegressor(
                n_estimators=config["n_estimators"],
                max_depth=config["max_depth"],
                max_features=config["max_features"],
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            ))
        ]
    )

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    train_mae = mean_absolute_error(y_train, train_pred)
    val_mae = mean_absolute_error(y_val, val_pred)

    results.append({
        "config": config,
        "train_mae": train_mae,
        "val_mae": val_mae,
        "model": model
    })

    print(
        f"Arbori={config['n_estimators']:3d} | "
        f"Depth={config['max_depth']:2d} | "
        f"Features={config['max_features']} | "
        f"Train MAE={train_mae:,.2f} | "
        f"Validation MAE={val_mae:,.2f}"
    )

# 7. Alegem configuratia folosind validation
best_result = min(results, key=lambda r: r["val_mae"])
best_config = best_result["config"]

print("\nCea mai buna configuratie:", best_config)
print(f"MAE Validation: {best_result['val_mae']:,.2f} EUR")

# Calculam importanta pe validation pentru modelul ales, antrenat doar pe train.
# O valoare pozitiva arata cu cat creste MAE cand amestecam acea caracteristica.
importance = permutation_importance(
    best_result["model"],
    X_val,
    y_val,
    scoring="neg_mean_absolute_error",
    n_repeats=5,
    random_state=42,
    n_jobs=-1
)

print("\nImportanta caracteristicilor pe validation (cresterea MAE):")
for feature, value in zip(features, importance.importances_mean):
    print(f"{feature}: {value:,.2f} EUR")

# 8. Reantrenam pe train + validation
final_model = Pipeline(
    steps=[
        ("preprocessor", clone(preprocessor)),
        ("regression", RandomForestRegressor(
            **best_config,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        ))
    ]
)

final_model.fit(X_temp, y_temp)

# 9. Evaluarea pe test ramane dezactivata pana finalizam selectia pe validation.
# test_pred = final_model.predict(X_test)
# test_mae = mean_absolute_error(y_test, test_pred)
# print(f"MAE final pe test: {test_mae:,.2f} EUR")

# 10. Testam predictia
apartment = pd.DataFrame([{
    "area_sqm": 70,
    "rooms": 3,
    "floor_number": 3,
    "total_floors": 10,
    "relative_floor": 3 / 10,
    "is_ground_floor": 0.0,
    "is_top_floor": 0.0,
    "area": "militari"
}])

prediction = final_model.predict(apartment)[0]

print(f"Pret estimat pentru 70 m² in Militari: {prediction:,.2f} EUR")
