import pandas as pd
from floor_features import extract_floor_features

df = pd.read_csv("data/bucharest_apartments_clean_stage1.csv")

df = extract_floor_features(df)

print("Etaje identificate:")
print(df["floor_number"].notna().mean() * 100, "%")

print("Numar total etaje identificat:")
print(df["total_floors"].notna().mean() * 100, "%")

print("\nExemple:")
print(
    df[["floor", "floor_number", "total_floors"]]
    .sample(20, random_state=42)
    .to_string(index=False)
)
