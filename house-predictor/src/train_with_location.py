
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# 1. Citim datele
df = pd.read_csv("data/bucharest_apartments_clean_stage1.csv")

# 2. Pastram aceleasi filtre ca la modelele anterioare
df = df[
    (df["area_sqm"] >= 20) &
    (df["area_sqm"] <= 200) &
    (df["price_eur"] >= 20000) &
    (df["price_eur"] <= 500000)
].copy()

# 3. Alegem caracteristicile
numeric_features = ["area_sqm", "rooms"]
categorical_features = ["area"]

features = numeric_features + categorical_features

X = df[features]
y = df["price_eur"]

# 4. Impartim train / test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 5. Pregatim caracteristicile numerice
numeric_transformer = SimpleImputer(strategy="median")

# 6. Pregatim localizarea
categorical_transformer = OneHotEncoder(
    handle_unknown="ignore"
)

# 7. Combinam transformarile
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# 8. Construim pipeline-ul complet
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regression", LinearRegression())
    ]
)

# 9. Antrenam modelul
model.fit(X_train, y_train)

# 10. Facem predictii
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# 11. Comparam eroarea pe datele de antrenare si pe cele de test.
train_mae = mean_absolute_error(y_train, train_predictions)
test_mae = mean_absolute_error(y_test, test_predictions)

print(f"MAE Train: {train_mae:,.2f} EUR")
print(f"MAE Test:  {test_mae:,.2f} EUR")

# 12. Testam un apartament nou
apartment = pd.DataFrame([
    {
        "area_sqm": 70,
        "rooms": 3,
        "area": "militari"
    }
])

estimated_price = model.predict(apartment)[0]

print(f"Pret estimat pentru 70 m² in Militari: {estimated_price:,.2f} EUR")
