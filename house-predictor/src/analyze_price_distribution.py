
import pandas as pd

df = pd.read_csv("data/bucharest_apartments_clean_stage1.csv")

df = df[
    df["area_sqm"].between(20, 200) &
    df["price_eur"].between(20000, 500000)
].copy()

# Pretul pe metru patrat, doar pentru analiza
df["price_per_sqm"] = df["price_eur"] / df["area_sqm"]

print("Numar total de apartamente:", len(df))

print("\nDistributia preturilor pe m2:")
print(
    df["price_per_sqm"].describe(
        percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    )
)

intervals = [
    (0, 1000),
    (1000, 1500),
    (1500, 2000),
    (2000, 2500),
    (2500, 3000),
    (3000, 3500),
    (3500, 4500),
    (4500, 6000),
    (6000, float("inf"))
]

print("\nApartamente pe intervale:")

for low, high in intervals:
    selected = df[
        (df["price_per_sqm"] >= low) &
        (df["price_per_sqm"] < high)
    ]

    percent = len(selected) / len(df) * 100

    print(
        f"{low:5.0f} - {high:7.0f} EUR/m2: "
        f"{len(selected):6d} apartamente "
        f"({percent:5.2f}%)"
    )
