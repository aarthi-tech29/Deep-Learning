from flask import Flask,render_template,request,jsonify
import random
import json

app = Flask(__name__)

with open("intents.json") as file:
    intents = json.load(file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_response():

    user_text = request.args.get("msg")

    for intent in intents["intents"]:

        for pattern in intent["patterns"]:

            if user_text.lower() in pattern.lower():

                return random.choice(
                    intent["responses"]
                )

    return "Sorry, I don't understand."

if __name__ == "__main__":
    app.run(debug=True)
    
# Input
# Greeting
# Hi
# Hello
# Hey
# Good morning
# Good evening

# About Company
# Who are you
# Tell me about company
# What is your company

# Courses
# What courses do you offer
# Available courses
# Course details

# Fees
# What is the fee
# Course fee
# Fees structure

# Contact
# Contact number
# Phone number
# How can I contact you

# Goodbye
# Bye
# Goodbye
# See you