
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error

from floor_features import extract_floor_features


# 1. Citim datasetul
df = pd.read_csv(
    "data/bucharest_apartments_clean_stage1.csv"
)

# 2. Aplicam aceleasi filtre ca la Random Forest
df = df[
    df["area_sqm"].between(20, 200)
    & df["price_eur"].between(20000, 500000)
].copy()

# 3. Generam caracteristicile despre etaj
df = extract_floor_features(df)

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


# 4. Train 60%, validation 20%, test 20%
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.25,
    random_state=42
)


configs = [
    {
        "n_estimators": 100,
        "max_depth": 20,
        "max_features": 1.0
    },
    {
        "n_estimators": 100,
        "max_depth": 30,
        "max_features": 0.7
    },
    {
        "n_estimators": 200,
        "max_depth": 30,
        "max_features": 0.7
    },
    {
        "n_estimators": 300,
        "max_depth": None,
        "max_features": 1.0
    }
]

best_mae = float("inf")
best_model = None
best_config = None

for config in configs:

    # Preprocesor nou pentru fiecare model
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                SimpleImputer(strategy="median"),
                numeric_features
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
            )
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                ExtraTreesRegressor(
                    **config,
                    min_samples_leaf=10,
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    train_mae = mean_absolute_error(
        y_train, train_pred
    )

    val_mae = mean_absolute_error(
        y_val, val_pred
    )

    print(
        f"Arbori={config['n_estimators']} | "
        f"Depth={config['max_depth']} | "
        f"Features={config['max_features']} | "
        f"Train MAE={train_mae:,.2f} | "
        f"Validation MAE={val_mae:,.2f}"
    )

    if val_mae < best_mae:
        best_mae = val_mae
        best_model = model
        best_config = config


print("\nCea mai buna configuratie:", best_config)
print(f"MAE Validation: {best_mae:,.2f} EUR")

print("\nComparatie cu Random Forest:")

rf_mae = 26936.72
difference = best_mae - rf_mae

print(f"Random Forest: {rf_mae:,.2f} EUR")
print(f"Extra Trees:   {best_mae:,.2f} EUR")

if difference < 0:
    print(
        f"Extra Trees este mai bun cu "
        f"{abs(difference):,.2f} EUR"
    )
else:
    print(
        f"Random Forest este mai bun cu "
        f"{difference:,.2f} EUR"
    )
