import pandas as pd

df = pd.read_csv(
    "dataset/ecommerce_data.csv"
)

search = input(
    "Search Product: "
)

results = df[

    df["Product"]
    .str.contains(
        search,
        case=False
    )

]

print(
    "\nSuggestions\n"
)

print(
    results["Product"]
    .unique()
)

# C001