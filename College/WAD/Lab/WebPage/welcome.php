<?php
// This script processes the form data

// collect value of input field
$name = $_POST['name'];
$phone = $_POST['phone'];
$email = $_POST['email'];
$concern = $_POST['concern'];

// You would then perhaps save these to a database or send an email

// Show confirmation or success message, or perhaps redirect to another page
echo "Thank you, $name. Your information has been received.";
