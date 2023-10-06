// Write down a program that includes all 3 types of prompts: alert, prompt, confirm.

var name = prompt("Enter your name: ");
document.write("Hello " + name);
var age = prompt("Enter your age: ");
document.write("<br/>Your age is " + age);
var gender = confirm("Are you sure Male?")
if (gender) {
    document.write("<br/>Your gender is male");
}
else {
    document.write("<br/>Your gender is female");
}
alert("Thank you for your information")