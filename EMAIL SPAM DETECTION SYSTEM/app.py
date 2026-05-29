from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(
    open("spam_model.pkl", "rb")
)

vectorizer = pickle.load(
    open("vectorizer.pkl", "rb")
)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    email = request.form["email"]

    transformed = vectorizer.transform([email])

    prediction = model.predict(transformed)

    result = prediction[0].upper()

    return render_template(
        "result.html",
        result=result,
        email=email
    )

if __name__ == "__main__":
    app.run(debug=True)
    
# Input
# Congratulations!

# You have won a FREE iPhone 17.

# Click the link below to claim your reward immediately.

# Offer expires in 24 hours.// SPAM
# //spam
# Limited Time Offer!

# Get 90% discount on all products.

# Buy now before the sale ends.// SPAM

# Not SPAM
# Hi Team,

# The project review meeting is scheduled for tomorrow at 10:00 AM.

# Please be present in the conference room.

# Thank you.
# Not SPAM
# Hello,

# Your order has been successfully delivered.

# Thank you for shopping with us.