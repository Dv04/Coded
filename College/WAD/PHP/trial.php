<?php

    $a = "Dev";
    echo "Hello ".$a;
    
    $r = 4;
    define("PI",3.1416);
    $area = PI * $r * $r;
    
    echo "\n\n".$area;
    
    $r1 = 100;
    function add ($r){
        global $r1;
        $sum = $r1 + $r;
        echo "\n\n".$sum;
    echo "\n";
    }
    
    add($r);
?>