from flask import Flask,render_template,request,jsonify

from chatbot import get_response

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get",methods=["POST"])
def chatbot_response():

    message = request.form["msg"]

    response = get_response(message)

    return jsonify(
        {"response":response}
    )

if __name__ == "__main__":
    app.run(debug=True)
    
# Hi
# What is your name?
# Can you help me?
# What is AI?
# What time is it?
# How is the weather?
# Tell me something interesting
# Thank you
# Bye