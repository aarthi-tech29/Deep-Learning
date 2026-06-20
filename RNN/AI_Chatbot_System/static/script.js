function sendMessage(){

let message =
document.getElementById(
"user-input"
).value;

fetch("/get",{
method:"POST",
headers:{
"Content-Type":
"application/x-www-form-urlencoded"
},
body:"msg="+message
})
.then(response=>response.json())
.then(data=>{

let box =
document.getElementById(
"chat-box"
);

box.innerHTML +=
"<p><b>You:</b> "
+ message +
"</p>";

box.innerHTML +=
"<p><b>Bot:</b> "
+ data.response +
"</p>";

});
}