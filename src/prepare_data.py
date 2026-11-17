import pandas as pd
from floor_features import extract_floor_features


# Coloanele dupa care consideram ca doua randuri sunt acelasi apartament.
# Acelasi anunt apare des repostat sub alt ID: aceeasi suprafata, camere,
# etaj, zona si exact acelasi pret.
DUPLICATE_KEY = ["area_sqm", "rooms", "floor", "area", "price_eur"]


def load_dataset(path="data/bucharest_apartments_clean_stage1.csv",
                 verbose=True):
    """Citeste datele, adauga caracteristicile de etaj, aplica filtrele
    si elimina anunturile duplicate.

    Toate scripturile de antrenare folosesc aceasta functie, ca sa
    compare modelele pe exact aceleasi date.
    """
    df = pd.read_csv(path)
    df = extract_floor_features(df)

    df = df[
        (df["area_sqm"] >= 20) &
        (df["area_sqm"] <= 200) &
        (df["price_eur"] >= 20000) &
        (df["price_eur"] <= 500000)
    ].copy()

    rows_before = len(df)

    # Eliminam duplicatele INAINTE de split. Altfel acelasi apartament
    # poate ajunge si in train si in validation/test, iar modelul il
    # "recunoaste" in loc sa-i estimeze pretul (data leakage).
    df = df.drop_duplicates(subset=DUPLICATE_KEY, keep="first")

    if verbose:
        removed = rows_before - len(df)
        print(
            f"Randuri dupa filtre: {rows_before:,} | "
            f"duplicate eliminate: {removed:,} "
            f"({100 * removed / rows_before:.1f}%) | "
            f"raman: {len(df):,}"
        )

    return df
