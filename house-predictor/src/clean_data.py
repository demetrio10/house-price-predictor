
import pandas as pd

# 1. Citim datasetul brut
df = pd.read_csv(
    "data/bucharest_apartments_raw.csv",
    dtype={"listing_id": "string", "site": "string"}
)

print("Inregistrari initiale:", len(df))

# 2. Verificam duplicatele
duplicates = df.duplicated(
    subset=["site", "listing_id"]
)

print("Inregistrari duplicate:", duplicates.sum())

# 3. Transformam data colectarii
df["scraped_at"] = pd.to_datetime(
    df["scraped_at"],
    errors="coerce",
    utc=True
)

# 4. Sortam cronologic
df = df.sort_values("scraped_at")

# 5. Eliminam duplicatele
df = df.drop_duplicates(
    subset=["site", "listing_id"],
    keep="last"
)

print("\nInregistrari dupa deduplicare:", len(df))

# 6. Salvam rezultatul deduplicarii
df.to_csv(
    "data/bucharest_apartments_deduplicated.csv",
    index=False
)

# 7. Analizam valorile lipsa
print("\nValori lipsa:")
print(df.isnull().sum())

missing_percent = df.isnull().mean() * 100

print("\nProcent valori lipsa:")
print(missing_percent.round(2))

# 8. Eliminam randurile fara camere sau suprafata
df = df.dropna(subset=["rooms", "area_sqm"])

print("\nInregistrari dupa eliminarea valorilor lipsa:")
print(len(df))

# 9. Verificam valorile invalide
print("\nPreturi <= 0:")
print((df["price_eur"] <= 0).sum())

print("\nSuprafete <= 0:")
print((df["area_sqm"] <= 0).sum())

print("\nCamere <= 0:")
print((df["rooms"] <= 0).sum())

# Eliminam valorile nepozitive. Preturile lipsa raman neschimbate.
invalid = (
    (df["price_eur"] <= 0)
    | (df["area_sqm"] <= 0)
    | (df["rooms"] <= 0)
)
df = df.loc[~invalid].copy()

print("\nInregistrari dupa eliminarea valorilor nepozitive:", len(df))

# Calculam pretul pe metru patrat dupa eliminarea suprafetelor nepozitive.
df["price_per_sqm"] = df["price_eur"] / df["area_sqm"]

# Investigam suprafetele neobisnuit de mari, fara sa eliminam randurile.
suspect = df[df["area_sqm"] > 1000].copy()

print("\nNumar apartamente cu suprafata peste 1000 mp:")
print(len(suspect))

print("\nExemple de valori suspecte:")
print(
    suspect[
        ["listing_id", "site", "area", "rooms",
         "area_sqm", "price_eur", "price_per_sqm"]
    ].head(20).to_string(index=False)
)

# 10. Afisam cele mai mari suprafete si cele mai mari preturi.
display_columns = ["listing_id", "area", "rooms", "area_sqm", "price_eur"]

print("\nTop 10 apartamente dupa suprafata:")
print(df.nlargest(10, "area_sqm")[display_columns].to_string(index=False))

print("\nTop 10 apartamente dupa pret:")
print(df.nlargest(10, "price_eur")[display_columns].to_string(index=False))

# 11. Calculam statisticile si marcam potentialele valori extreme.
# Metoda IQR semnaleaza valori pentru verificare, fara sa le elimine.
for column in ["price_eur", "area_sqm"]:
    print(f"\nStatistici descriptive pentru {column}:")
    print(df[column].describe())

    q1, q2, q3 = df[column].quantile([0.25, 0.50, 0.75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    print(f"Q1: {q1:.2f}, Q2 (mediana): {q2:.2f}, Q3: {q3:.2f}")
    print(f"IQR (Q3 - Q1): {iqr:.2f}")
    print(f"Limite IQR: [{lower_bound:.2f}, {upper_bound:.2f}]")

    # Pastram marcajele si in CSV; valorile lipsa nu sunt marcate ca outlieri.
    outlier_column = f"{column}_iqr_outlier"
    df[outlier_column] = (
        (df[column] < lower_bound) | (df[column] > upper_bound)
    )
    print("Potentiali outlieri (pastrati):", df[outlier_column].sum())

# 12. Salvam separat; datasetul brut si cel deduplicat isi pastreaza rolul.
df.to_csv(
    "data/bucharest_apartments_clean_stage1.csv",
    index=False
)
