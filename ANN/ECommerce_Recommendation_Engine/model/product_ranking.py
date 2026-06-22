import pandas as pd

df = pd.read_csv(
    "dataset/ecommerce_data.csv"
)

df["Score"] = (

    df["PurchaseHistory"] * 5

    +

    df["CartItems"] * 3

    +

    df["Clicks"]

)

ranking = df.groupby(
    "Product"
)["Score"].sum()

ranking = ranking.sort_values(
    ascending=False
)

print(
    "\nTop Ranked Products\n"
)

print(
    ranking.head(10)
)

# C001