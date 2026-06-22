import pandas as pd

df = pd.read_csv(
    "dataset/ecommerce_data.csv"
)

customer_id = input(
    "Enter Customer ID: "
)

customer = df[
    df["CustomerID"] == customer_id
]

if customer.empty:

    print("Customer Not Found")

else:

    total_searches = len(
        customer
    )

    total_purchases = customer[
        "PurchaseHistory"
    ].sum()

    total_cart = customer[
        "CartItems"
    ].sum()

    total_clicks = customer[
        "Clicks"
    ].sum()

    print("\nCustomer Analytics\n")

    print(
        "Search Records:",
        total_searches
    )

    print(
        "Purchases:",
        total_purchases
    )

    print(
        "Cart Items:",
        total_cart
    )

    print(
        "Clicks:",
        total_clicks
    )
    
# C001