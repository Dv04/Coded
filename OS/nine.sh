echo "Main Menu"
echo "Press 1 for using Grep Command"
echo "Press 2 for using Egrep Command"
echo "Press 3 for using Fgrep Command"
read -p "Enter your choice : " a

case $a in
                1) read -p "For single pattern search, Enter Pattern below : " b
                   grep "$b" null.txt
                   ;;
                2) read -p "For double Pattern search, Enter b,c pattern : " b c
                   egrep "$b" null.txt
                   grep -E "$c" null.txt
                   ;;
                3) read -p "For Pattern From a File, Enter Pattern : " b
                   grep -F "$b" null.txt
                   ;;

esac