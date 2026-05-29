from flask import Flask,render_template,request
import pickle

app = Flask(__name__)

model = pickle.load(
    open("sentiment_model.pkl","rb")
)

vectorizer = pickle.load(
    open("vectorizer.pkl","rb")
)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():

    review = request.form["review"]

    transformed = vectorizer.transform([review])

    prediction = model.predict(transformed)

    result = prediction[0].upper()

    return render_template(
        "result.html",
        result=result,
        review=review
    )

if __name__ == "__main__":
    app.run(debug=True)
    
# Input
# This movie was fantastic. The acting was brilliant and the story was amazing.// POSITIVE
# This was one of the worst movies I have ever watched. Waste of time.// NEGATIVE