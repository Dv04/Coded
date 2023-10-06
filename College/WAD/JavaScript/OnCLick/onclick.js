// Get the button elements
const button1 = document.getElementById("button1");
const button2 = document.getElementById("button2");
const button3 = document.getElementById("button3");
const button4 = document.getElementById("button4");
const button5 = document.getElementById("button5");
const button6 = document.getElementById("button6");

// Define the onclick event handlers
button1.onclick = function () {
    console.log("Button 1 clicked");
};

button2.onclick = function () {
    alert("Button 2 clicked");
};

button3.onclick = function () {
    document.body.style.backgroundColor = "red";
};

button4.onclick = function () {
    document.body.style.backgroundColor = "green";
};

button5.onclick = function () {
    document.body.style.backgroundColor = "blue";
};

button6.onclick = function () {
    document.body.style.backgroundColor = "white";
};

// Define the onmouseover event handlers
button1.onmouseover = function () {
    console.log("Mouse over button 1");
};

button2.onmouseover = function () {
    // alert("Mouse over button 2");
};

button3.onmouseover = function () {
    document.body.style.color = "red";
};

button4.onmouseover = function () {
    document.body.style.color = "green";
};

button5.onmouseover = function () {
    document.body.style.color = "blue";
};

button6.onmouseover = function () {
    document.body.style.color = "white";
};

// Define the onmouseout event handlers
button1.onmouseout = function () {
    console.log("Mouse out button 1");
};

button2.onmouseout = function () {
    alert("Mouse out button 2");
};

button3.onmouseout = function () {
    document.body.style.color = "black";
};

button4.onmouseout = function () {
    document.body.style.color = "black";
};

button5.onmouseout = function () {
    document.body.style.color = "black";
};

button6.onmouseout = function () {
    document.body.style.color = "black";
};

// Define the onmousedown event handlers
button1.onmousedown = function () {
    console.log("Mouse down button 1");
};

button2.onmousedown = function () {
    alert("Mouse down button 2");
};

button3.onmousedown = function () {
    document.body.style.fontSize = "20px";
};

button4.onmousedown = function () {
    document.body.style.fontSize = "30px";
};

button5.onmousedown = function () {
    document.body.style.fontSize = "40px";
};

button6.onmousedown = function () {
    document.body.style.fontSize = "50px";
};

// Define the onmouseup event handlers
button1.onmouseup = function () {
    console.log("Mouse up button 1");
};

button2.onmouseup = function () {
    alert("Mouse up button 2");
};

button3.onmouseup = function () {
    document.body.style.fontSize = "16px";
};

button4.onmouseup = function () {
    document.body.style.fontSize = "18px";
};

button5.onmouseup = function () {
    document.body.style.fontSize = "22px";
};

button6.onmouseup = function () {
    document.body.style.fontSize = "24px";
};

// Define the ondblclick event handlers
button1.ondblclick = function () {
    console.log("Double click button 1");
};

button2.ondblclick = function () {
    alert("Double click button 2");
};

button3.ondblclick = function () {
    document.body.style.fontWeight = "bold";
};

button4.ondblclick = function () {
    document.body.style.fontWeight = "normal";
};

button5.ondblclick = function () {
    document.body.style.fontStyle = "italic";
};

button6.ondblclick = function () {
    document.body.style.fontStyle = "normal";
};
