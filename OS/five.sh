# 1) Display calendar of current month.
# 2) Display today’s date and time.
# 3) Display usernames those are currently logged in the system.
# 4) Display your name at given x, y position.
# 5) Display your terminal number.

echo -e "\nWelcome to the Menu\n1. Display calendar of current month\n2. Display today’s date and time\n3. Display usernames those are currently logged in the system\n4. Display your name at given x, y position\n5. Display your terminal number\n6. Exit"
read -p "Enter your choice: " choice

while [ $choice -ne 6 ]
do
    case $choice in
        1) cal;;
        2) date;;
        3) who;;
        4) read -p "Enter x and y coordinates: " x y
           tput cup $x $y
           read -p "Enter your name: " name
           echo $name;;
        5) tty;;
        6) exit;;
        *) echo "Invalid choice";;
    esac
    read -p "Enter your choice: " choice
done