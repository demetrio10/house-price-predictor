import re
import pandas as pd


def extract_floor_features(df):
    df = df.copy()

    floor_numbers = []
    total_floors = []

    for value in df["floor"]:
        text = str(value).lower().strip()

        floor = None
        total = None

        # Cazul: "Etaj 2 / 10", "etaj 3/4"
        match = re.search(
            r"etaj\s*(\d+)\s*/\s*(\d+)",
            text
        )

        if match:
            floor = int(match.group(1))
            total = int(match.group(2))

        # Cazul: "Parter / 4"
        elif re.search(r"\bparter\b", text):
            floor = 0

            match_total = re.search(
                r"\bparter\s*/\s*(\d+)",
                text
            )

            if match_total:
                total = int(match_total.group(1))

        # Cazul: "Etaj 2", "etaj 5, bloc..."
        else:
            match = re.search(r"\betaj\s*(\d+)\b", text)

            if match:
                floor = int(match.group(1))

            # Cazul: "1", "2", "3"
            elif re.fullmatch(r"\d+", text):
                floor = int(text)

        floor_numbers.append(floor)
        total_floors.append(total)

    df["floor_number"] = floor_numbers
    df["total_floors"] = total_floors

    # Pozitia relativa in cladire
    df["relative_floor"] = (
        df["floor_number"] / df["total_floors"]
    )

    # Eliminam rapoartele imposibile sau nedefinite
    invalid = (
        df["total_floors"].isna()
        | (df["total_floors"] <= 0)
        | df["floor_number"].isna()
        | (df["floor_number"] < 0)
        | (df["floor_number"] > df["total_floors"])
    )

    df.loc[invalid, "relative_floor"] = float("nan")

    # Indicator pentru parter
    df["is_ground_floor"] = (
        df["floor_number"] == 0
    ).astype(float)

    df.loc[
        df["floor_number"].isna(),
        "is_ground_floor"
    ] = float("nan")

    # Indicator pentru ultimul etaj
    df["is_top_floor"] = (
        df["floor_number"] == df["total_floors"]
    ).astype(float)

    df.loc[invalid, "is_top_floor"] = float("nan")

    return df
