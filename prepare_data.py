import pandas as pd

import config


ALLOWED_DEPARTMENTS = [
    "European Paintings",
    "European Sculpture and Decorative Arts",
    "The American Wing",
    "Greek and Roman Art",
    "Egyptian Art",
    "Islamic Art",
    "Asian Art",
    "Arms and Armor"
]


def main():

    print("Loading Met Museum dataset...")

    df = pd.read_csv(
        config.MET_CSV_URL,
        low_memory=False
    )

    print("Total rows:", len(df))

    public_domain = (
        df["Is Public Domain"]
        .astype(str)
        .str.upper()
        .isin(["TRUE", "1", "YES"])
    )

    df = df[
        public_domain
    ].copy()

    df = df[
        df["Department"].isin(
            ALLOWED_DEPARTMENTS
        )
    ]

    print(
        "Filtered rows:",
        len(df)
    )

    samples = []

    for dept, group in df.groupby(
        "Department"
    ):

        n = min(
            config.SAMPLE_PER_DEPARTMENT,
            len(group)
        )

        sampled = group.sample(
            n=n,
            random_state=42
        )

        samples.append(
            sampled
        )

        print(
            f"{dept}: {len(sampled)}"
        )

    df = pd.concat(
        samples,
        ignore_index=True
    )

    keep_cols = [
        "Object ID",
        "Title",
        "Department",
        "Classification",
        "Medium",
        "Object Date"
    ]

    df = df[
        keep_cols
    ].copy()

    df.columns = [
        "object_id",
        "title",
        "department",
        "classification",
        "medium",
        "object_date"
    ]

    df.to_csv(
        config.METADATA_PATH,
        index=False
    )

    print(
        f"\nSaved {len(df)} rows -> "
        f"{config.METADATA_PATH}"
    )


if __name__ == "__main__":
    main()
