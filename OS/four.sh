# Write a menu driven shell script which will print the following menu and execute the given task.

# 1. Display the current date and time
# 2. Display the username
# 3. Display the current month calendar
# 4. Display all the files in the current directory
# 5. Exit

# The script should run until the user selects the exit option.

echo -e "\nWelcome to the Menu\n1. Display the current date and time\n2. Display the username\n3. Display the current month calendar\n4. Display all the files in the current directory\n5. Exit"

read -p "Enter your choice: " choice

while [ $choice -ne 5 ]
do
    case $choice in
        1) date;;
        2) whoami;;
        3) cal;;
        4) ls;;
        5) exit;;
        *) echo "Invalid choice";;
    esac
    read -p "Enter your choice: " choice
done