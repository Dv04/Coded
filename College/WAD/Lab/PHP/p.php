<!DOCTYPE html>
<html>

<head>
    <title>Submitted Data (GET)</title>
</head>

<body>
    <h1>Submitted Data (GET)</h1>
    <?php
    $name = $_GET["name_get"];
    $age = $_GET["age_get"];
    ?>
    <p>Name: <?php echo $name; ?></p>
    <p>Age: <?php echo $age; ?></p>
</body>

</html>