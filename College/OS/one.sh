# Write a shell script to generate marksheet of a student. Take 3 subjects, calculate and display total marks, percentage and Class obtained by the student.

echo -e "\nWelcome"
echo "Enter your name: "
read name
echo "Enter the following marks out of 100"
echo "Enter your marks for OS: "
read os
echo "Enter your marks for DM: "
read dm
echo "Enter your marks for DCN: "
read dcn

total=`expr $os + $dm + $dcn`
percentage=`expr $total / 3`

echo "Name: $name"
echo "Total marks: $total"
echo "Percentage: $percentage"

if [ $percentage -ge 90 ]
then
    echo "Class: Distinction"
elif [ $percentage -ge 80 ]
then
    echo "Class: First Class"
elif [ $percentage -ge 70 ]
then
    echo "Class: Second Class"
elif [ $percentage -ge 60 ]
then
    echo "Class: Third Class"
else
    echo "Class: Fail"
fi