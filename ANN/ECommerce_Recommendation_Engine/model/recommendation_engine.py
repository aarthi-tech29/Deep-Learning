import pandas as pd

df = pd.read_csv(
    "dataset/ecommerce_data.csv"
)

customer_id = input(
    "Enter Customer ID: "
)

customer_data = df[
    df["CustomerID"] == customer_id
]

if customer_data.empty:

    print("Customer Not Found")

else:

    category = customer_data[
        "Category"
    ].mode()[0]

    recommendations = df[

        (df["Category"] == category)

        &

        (~df["Product"].isin(
            customer_data["Product"]
        ))

    ]

    print("\nRecommended Products\n")

    for product in recommendations[
        "Product"
    ].unique():

        print(product)
        
# C001