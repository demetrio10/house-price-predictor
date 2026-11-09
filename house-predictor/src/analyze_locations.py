
import pandas as pd

# 1. Citim datasetul curatat
df = pd.read_csv("data/bucharest_apartments_clean_stage1.csv")

# 2. Aplicam aceleasi filtre ca la antrenarea modelului
df = df[
    (df["area_sqm"] >= 20) &
    (df["area_sqm"] <= 200) &
    (df["price_eur"] >= 20000) &
    (df["price_eur"] <= 500000)
].copy()

# 3. Afisam numarul total de apartamente
print("Total apartamente:", len(df))

# 4. Afisam cele mai frecvente 30 de zone
print("\nCele mai frecvente 30 de zone:")
print(df["area"].value_counts().head(30))

# 5. Numarul de zone distincte
print("\nNumar zone distincte:")
print(df["area"].nunique())

# 6. Numarul valorilor lipsa
print("\nZone lipsa:")
print(df["area"].isna().sum())

# 7. Procentul apartamentelor fara zona
missing_percent = df["area"].isna().mean() * 100

print(f"\nProcent zone lipsa: {missing_percent:.2f}%")

# 8. Pretul median pentru fiecare zona
median_prices = (
    df.groupby("area")["price_eur"]
    .agg(["count", "median"])
    .sort_values("count", ascending=False)
)

print("\nStatistici pentru cele mai frecvente 20 de zone:")
print(median_prices.head(20))
