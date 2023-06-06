function capitalize(s) {
    return toupper(substr(s, 1, 1)) substr(s, 2)
}

{
    for (i = 1; i <= NF; i++) {
        $i = capitalize($i)
    }
    print
}