import pandas as pd

df = pd.read_csv(
    "data/valid_metadata_blip.csv"
)

print(
    df[["title", "blip_caption"]]
    .head(10)
)