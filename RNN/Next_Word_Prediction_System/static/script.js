async function predictWord(){

    let text = document
    .getElementById(
        "text"
    )
    .value;

    if(
        text.trim() === ""
    ){
        alert(
            "Enter text"
        );
        return;
    }

    let response =
    await fetch(
        "/predict",
        {
            method:"POST",

            headers:{
                "Content-Type":
                "application/json"
            },

            body:JSON.stringify({
                text:text
            })
        }
    );

    let data =
    await response.json();

    document.getElementById("result").innerHTML =
    "<span class='prediction'>🔮 Next Word: "
    + data.word +
    "</span>";}