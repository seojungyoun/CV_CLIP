import pandas as pd
import config

df = pd.read_csv(config.METADATA_PATH)

print(df["department"].value_counts().head(20))