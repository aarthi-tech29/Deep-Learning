async function detectSpam() {

    const emailText =
        document.getElementById(
            "emailInput"
        ).value;

    const resultDiv =
        document.getElementById(
            "result"
        );

    const tagsDiv =
        document.getElementById(
            "tags"
        );

    if (
        emailText.trim() === ""
    ) {

        resultDiv.innerHTML =
            "⚠️ Please enter an email";

        resultDiv.className =
            "result";

        tagsDiv.innerHTML = "";

        return;
    }

    resultDiv.innerHTML =
        "⏳ Analyzing Email...";

    tagsDiv.innerHTML = "";

    try {

        const response =
            await fetch(
                "/predict",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        email: emailText
                    })
                }
            );

        const data =
            await response.json();

        resultDiv.innerHTML =
            data.prediction;

        if (
            data.prediction.includes(
                "SPAM"
            )
        ) {

            resultDiv.className =
                "result spam";
        }
        else {

            resultDiv.className =
                "result ham";
        }

        tagsDiv.innerHTML =
            "<b>POS Tags:</b><br><br>" +
            data.tags;

    }
    catch (error) {

        resultDiv.innerHTML =
            "❌ Server Error";

        resultDiv.className =
            "result";

        tagsDiv.innerHTML =
            error;
    }
}