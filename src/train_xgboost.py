import pandas as pd
from prepare_data import load_dataset

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# 1-2. Citim datele, aplicam filtrele si eliminam duplicatele
# (aceeasi functie ca la Random Forest, ca sa putem compara corect)
df = load_dataset()

# 3. Aceleasi caracteristici ca la Random Forest
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

# 4. Acelasi split 60 / 20 / 20 cu random_state=42
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
# Diferenta fata de Random Forest: NU mai folosim SimpleImputer.
# XGBoost stie sa trateze singur valorile lipsa (NaN): la fiecare split
# invata in ce parte sa trimita apartamentele fara etaj cunoscut.
#
# sparse_output=False: XGBoost considera "lipsa" orice valoare nestocata
# intr-o matrice rara (sparse). Asta ar confunda 0 (ex. is_ground_floor=0)
# cu NaN, asa ca lucram cu o matrice densa.
def create_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features
            )
        ]
    )


# Early stopping are nevoie de validation deja transformat, deci ajustam
# preprocesorul DOAR pe train si transformam validation cu el.
preprocessor = create_preprocessor()
X_train_prep = preprocessor.fit_transform(X_train)
X_val_prep = preprocessor.transform(X_val)


# 6. Configuratii XGBoost
# n_estimators este doar o limita maxima: early stopping opreste
# antrenarea cand MAE pe validation nu mai scade 100 de runde la rand.
configs = [
    {"learning_rate": 0.1, "max_depth": 4, "min_child_weight": 1,
     "subsample": 1.0, "colsample_bytree": 1.0, "reg_lambda": 1.0,
     "objective": "reg:squarederror"},
    {"learning_rate": 0.1, "max_depth": 6, "min_child_weight": 1,
     "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0,
     "objective": "reg:squarederror"},
    {"learning_rate": 0.1, "max_depth": 8, "min_child_weight": 5,
     "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0,
     "objective": "reg:squarederror"},
    {"learning_rate": 0.1, "max_depth": 8, "min_child_weight": 10,
     "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 5.0,
     "objective": "reg:squarederror"},
    # Aceeasi configuratie, dar optimizeaza direct eroarea absoluta (MAE)
    {"learning_rate": 0.1, "max_depth": 8, "min_child_weight": 5,
     "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0,
     "objective": "reg:absoluteerror"},
]

results = []

for config in configs:
    model = XGBRegressor(
        **config,
        n_estimators=3000,
        early_stopping_rounds=100,
        eval_metric="mae",
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train_prep, y_train,
        eval_set=[(X_val_prep, y_val)],
        verbose=False
    )

    # Cate runde (arbori) au fost de fapt utile
    best_trees = model.best_iteration + 1

    train_mae = mean_absolute_error(y_train, model.predict(X_train_prep))
    val_mae = mean_absolute_error(y_val, model.predict(X_val_prep))

    results.append({
        "config": config,
        "n_trees": best_trees,
        "train_mae": train_mae,
        "val_mae": val_mae,
        "model": model
    })

    print(
        f"LR={config['learning_rate']:.2f} | "
        f"Depth={config['max_depth']} | "
        f"MCW={config['min_child_weight']:2d} | "
        f"Lambda={config['reg_lambda']:.0f} | "
        f"{config['objective']:17s} | "
        f"Arbori={best_trees:4d} | "
        f"Train MAE={train_mae:,.2f} | "
        f"Validation MAE={val_mae:,.2f}"
    )


# 7. Alegem configuratia folosind validation
best_result = min(results, key=lambda r: r["val_mae"])
best_config = best_result["config"]
best_trees = best_result["n_trees"]

print("\nCea mai buna configuratie:", best_config)
print(f"Numar de arbori ales prin early stopping: {best_trees}")
print(f"MAE Validation XGBoost: {best_result['val_mae']:,.2f} EUR")


# 8. Importanta caracteristicilor (gain = cat reduce in medie pierderea
# un split pe acea caracteristica). Coloanele one-hot pentru zone le
# adunam intr-o singura caracteristica "area".
feature_names = preprocessor.get_feature_names_out()
gain = best_result["model"].get_booster().get_score(importance_type="total_gain")

importance = {}
for i, name in enumerate(feature_names):
    value = gain.get(f"f{i}", 0.0)
    short = "area" if name.startswith("cat__") else name.replace("num__", "")
    importance[short] = importance.get(short, 0.0) + value

total = sum(importance.values())
print("\nImportanta caracteristicilor (procent din gain total):")
for name, value in sorted(importance.items(), key=lambda kv: -kv[1]):
    print(f"{name:16s} {100 * value / total:5.1f}%")


# 9. Reantrenam pe train + validation.
# Nu mai avem un set separat pentru early stopping, asa ca folosim
# numarul de arbori gasit mai sus.
final_model = Pipeline(
    steps=[
        ("preprocessor", create_preprocessor()),
        ("regression", XGBRegressor(
            **best_config,
            n_estimators=best_trees,
            tree_method="hist",
            random_state=42,
            n_jobs=-1
        ))
    ]
)

final_model.fit(X_temp, y_temp)

# 10. Evaluarea pe test ramane dezactivata pana finalizam selectia pe validation.
# test_pred = final_model.predict(X_test)
# test_mae = mean_absolute_error(y_test, test_pred)
# print(f"MAE final pe test: {test_mae:,.2f} EUR")

# 11. Testam predictia (acelasi apartament ca la Random Forest)
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

print(f"\nPret estimat pentru 70 m² in Militari: {prediction:,.2f} EUR")
