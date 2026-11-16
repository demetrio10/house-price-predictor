
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

# Datasetul de pe Hugging Face
DATASET_NAME = "palerflavius/ro-real-estate-listings"

# Coloanele necesare proiectului
COLUMNS = [
    "listing_id",
    "site",
    "city",
    "area",
    "offer_type",
    "property_type",
    "rooms",
    "area_sqm",
    "floor",
    "price_eur",
    "scraped_at",
]



print("Cautam fisierele pe Hugging Face...")

api = HfApi()
fs = HfFileSystem()

# Obtinem lista fisierelor din repository
files = api.list_repo_files(
    repo_id=DATASET_NAME,
    repo_type="dataset"
)

# Selectam doar fisierele relevante
selected_files = []

for file in files:
    if (
        file.endswith(".parquet")
        and (
            "county=bucuresti/" in file.lower()
            or "city=bucuresti/" in file.lower()
        )
        and "/offer=sale/" in file.lower()
        and "/property=apartments/" in file.lower()
    ):
        selected_files.append(file)

print("Fisiere relevante gasite:", len(selected_files))

if not selected_files:
    raise RuntimeError(
        "Nu am gasit fisiere pentru apartamente "
        "de vanzare in Bucuresti. "
        "Trebuie verificate valorile din cai."
    )

apartments = []

# Citim un numar limitat de fisiere pentru test
for index, file in enumerate(selected_files, start=1):
    print(f"Procesam fisierul {index}: {file}")

    # Citim direct de pe Hub, fara cai locale lungi in cache-ul Windows.
    remote_path = f"datasets/{DATASET_NAME}/{file}"
    with fs.open(remote_path, "rb") as remote_file:
        parquet_file = pq.ParquetFile(remote_file)

        # Citim numai coloanele disponibile si necesare.
        available_columns = [
            column for column in COLUMNS
            if column in parquet_file.schema_arrow.names
        ]

        table = parquet_file.read(columns=available_columns)

    df_part = table.to_pandas()

    # Verificam orasul si tipul anuntului
    if "city" in df_part.columns:
        cities = (
            df_part["city"].astype("string")
            .str.lower()
            .str.strip()
        )
        df_part = df_part[
            cities.isin(["bucuresti", "bucurești", "bucharest"])
        ]

    # Pastram doar inregistrarile filtrate
    if not df_part.empty:
        apartments.append(df_part)

if not apartments:
    raise RuntimeError(
        "Fisierele au fost citite, dar nu am gasit "
        "apartamente din Bucuresti in esantion."
    )

# Combinam datele intr-un singur tabel
df = pd.concat(
    apartments,
    ignore_index=True
)

# Completam coloanele care lipsesc
df = df.reindex(columns=COLUMNS)

# Verificam numarul inregistrarilor pe sectoare
print("\nInregistrari pe sectoare:")
print(df["area"].value_counts())

# Verificam cate identificatoare unice avem
print("\nNumar de listing_id unice:")
print(df["listing_id"].nunique())

# Verificam inregistrarile repetate
print("\nInregistrari cu listing_id repetat:")
print(df.duplicated(subset=["site", "listing_id"]).sum())

# Verificam intervalul de colectare
dates = pd.to_datetime(df["scraped_at"], errors="coerce", utc=True)

print("\nPrima colectare:", dates.min())
print("Ultima colectare:", dates.max())


# Cream folderul data
Path("data").mkdir(exist_ok=True)

# Salvam rezultatul
output = "data/bucharest_apartments_raw.csv"

df.to_csv(output, index=False)

print("\nDate salvate:", output)
print("Numar inregistrari:", len(df))
print("\nPrimele 5 randuri:")
print(df.head().to_string(index=False))
print("\nValori lipsa:")
print(df.isnull().sum())
