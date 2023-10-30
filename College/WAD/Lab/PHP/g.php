<!DOCTYPE html>
<html>

<head>
    <title>Submitted Data (POST)</title>
</head>

<body>
    <h1>Submitted Data (POST)</h1>
    <?php
    $name = $_POST["name_post"];
    $age = $_POST["age_post"];
    ?>
    <p>Name: <?php echo $name; ?></p>
    <p>Age: <?php echo $age; ?></p>
</body>

</html>