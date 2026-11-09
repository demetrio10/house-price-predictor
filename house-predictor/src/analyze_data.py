
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
    
# 1. Citim datasetul curatat
df = pd.read_csv(
    "data/bucharest_apartments_clean_stage1.csv"
)

# Calculam asimetria preturilor din intregul dataset curatat.
print("\nSkewness preturi:")
print(df["price_eur"].skew())

# 2. Selectam apartamentele cu preturi sub 500.000 EUR
df_plot = df[
    (df["price_eur"] > 0) &
    (df["price_eur"] <= 500000)
]

# 3. Cream histograma
plt.figure(figsize=(12, 6))

sns.histplot(
    data=df_plot,
    x="price_eur",
    bins=50
)

# 4. Adaugam titlul si etichetele
plt.title("Distributia preturilor apartamentelor din Bucuresti")
plt.xlabel("Pretul apartamentului (EUR)")
plt.ylabel("Numar de apartamente")

# 5. Selectam datele pentru scatter plot si corelatia Pearson.
df_scatter = df[
    (df["area_sqm"] >= 20) &
    (df["area_sqm"] <= 200) &
    (df["price_eur"] >= 20000) &
    (df["price_eur"] <= 500000)
].copy()

# Verificam daca filtrarea a functionat.
print("Inregistrari pentru scatter plot:", len(df_scatter))
print("Suprafata maxima:", df_scatter["area_sqm"].max())
print("Pret maxim:", df_scatter["price_eur"].max())

# Pearson masoara relatia liniara dintre suprafata si pret.
# Folosim acelasi esantion filtrat ca in scatter plot, nu intregul dataset.
pearson = df_scatter["area_sqm"].corr(
    df_scatter["price_eur"], method="pearson"
)
print(f"\nCorelatia Pearson suprafata-pret (date filtrate): {pearson:.4f}")

# 6. Cream al doilea grafic: fiecare punct reprezinta un anunt.
plt.figure(figsize=(12, 6))

sns.scatterplot(
    data=df_scatter,
    x="area_sqm",
    y="price_eur",
    alpha=0.15,
    s=10
)

plt.title("Relatia dintre suprafata si pret - Bucuresti")
plt.xlabel("Suprafata (m²)")
plt.ylabel("Pret (EUR)")
plt.grid(alpha=0.2)

# 7. Afisam ambele grafice.
plt.show()
