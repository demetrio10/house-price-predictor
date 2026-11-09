
import pandas as pd

df = pd.read_csv("data/bucharest_apartments_clean_stage1.csv")

print("Numar inregistrari:", len(df))

print("\nColoanele disponibile:")
for col in df.columns:
    missing = df[col].isna().mean() * 100
    unique = df[col].nunique(dropna=True)

    print(
        f"{col:25s} | "
        f"Tip: {str(df[col].dtype):10s} | "
        f"Lipsa: {missing:6.2f}% | "
        f"Valori distincte: {unique}"
    )

print("\nExemple din dataset:")
print(df.head(5).to_string(index=False))
