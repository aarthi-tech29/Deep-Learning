from flask import Flask,render_template,request
import pdfplumber
import pickle
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

model = pickle.load(
    open("resume_model.pkl","rb")
)

vectorizer = pickle.load(
    open("vectorizer.pkl","rb")
)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():

    jd = request.form["job_description"]

    resume = request.files["resume"]

    text = ""

    with pdfplumber.open(resume) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""

    resume_vector = vectorizer.transform([text])
    jd_vector = vectorizer.transform([jd])

    similarity = cosine_similarity(
        resume_vector,
        jd_vector
    )[0][0]

    score = round(similarity*100,2)

    result = "SELECTED ✅" if score >= 50 else "REJECTED ❌"

    return render_template(
        "result.html",
        score=score,
        result=result
    )

if __name__ == "__main__":
    app.run(debug=True)
    
# Input
# Job Description
# Looking for a Python Developer with Flask, SQL, Machine Learning, Pandas and NumPy skills.//John

# Job Description
# Looking for a Data Analyst.

# Required Skills:
# SQL
# Power BI
# Excel
# Tableau
# Statistics

# Experience:
# 0-2 Years // Emily Davis