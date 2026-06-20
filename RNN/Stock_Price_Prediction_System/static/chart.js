let stockChart = null;

const predictBtn =
document.getElementById(
"predictBtn"
);

predictBtn.addEventListener(
"click",
predictStock
);

async function predictStock(){

    const symbol =
    document
    .getElementById("symbol")
    .value
    .trim()
    .toUpperCase();

    if(symbol === ""){

        alert(
        "Please enter a stock symbol."
        );

        return;
    }

    predictBtn.disabled = true;

    predictBtn.innerHTML =
    "Predicting...";

    try{

        const response =
        await fetch(
            "/predict",
            {
                method:"POST",

                headers:{
                    "Content-Type":
                    "application/json"
                },

                body:JSON.stringify({
                    symbol:symbol
                })
            }
        );

        const data =
        await response.json();

        if(data.error){

            alert(data.error);

            predictBtn.disabled = false;

            predictBtn.innerHTML =
            "Predict";

            return;
        }

        // ====================
        // KPI CARDS
        // ====================

        document
        .getElementById(
        "currentPrice"
        )
        .innerHTML =
        "$" +
        data.current_price;

        document
        .getElementById(
        "predictedPrice"
        )
        .innerHTML =
        "$" +
        data.predicted_price;

        document
        .getElementById(
        "trend"
        )
        .innerHTML =
        data.trend;

        // ====================
        // CHART DATA
        // ====================

        const actualLabels =
        [];

        for(
            let i=1;
            i<=data.actual.length;
            i++
        ){

            actualLabels.push(
            "D"+i
            );
        }

        const predictionLabels =
        [
        "F1",
        "F2",
        "F3",
        "F4",
        "F5",
        "F6",
        "F7"
        ];

        const labels =
        [
            ...actualLabels,
            ...predictionLabels
        ];

        const actualSeries =
        [
            ...data.actual,
            null,
            null,
            null,
            null,
            null,
            null,
            null
        ];

        const predictedSeries =
        [
            ...Array(
            data.actual.length
            ).fill(null),

            ...data.predicted
        ];

        // ====================
        // DESTROY OLD CHART
        // ====================

        if(stockChart){

            stockChart.destroy();
        }

        const ctx =
        document
        .getElementById(
        "stockChart"
        );

        stockChart =
        new Chart(ctx,{

            type:"line",

            data:{

                labels:labels,

                datasets:[

                {
                    label:
                    "Actual Price",

                    data:
                    actualSeries,

                    borderColor:
                    "#38bdf8",

                    borderWidth:3,

                    tension:0.4
                },

                {
                    label:
                    "Predicted Price",

                    data:
                    predictedSeries,

                    borderColor:
                    "#22c55e",

                    borderWidth:3,

                    tension:0.4
                }

                ]
            },

            options:{

                responsive:true,

                maintainAspectRatio:
                false,

                plugins:{

                    legend:{

                        labels:{
                            color:
                            "white"
                        }
                    }
                },

                scales:{

                    x:{

                        ticks:{
                            color:
                            "white"
                        }
                    },

                    y:{

                        ticks:{
                            color:
                            "white"
                        }
                    }
                }
            }
        });

    }

    catch(error){

        console.log(error);

        alert(
        "Prediction failed."
        );
    }

    predictBtn.disabled = false;

    predictBtn.innerHTML =
    "Predict";
}