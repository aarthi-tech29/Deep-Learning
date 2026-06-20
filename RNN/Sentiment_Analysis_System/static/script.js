document
.getElementById(
"analyzeBtn"
)
.addEventListener(
"click",
async function(){

const review =
document
.getElementById(
"review"
).value;

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
review:review
})
}
);

const data =
await response.json();

document
.getElementById(
"result"
)
.innerHTML =
data.sentiment;
});