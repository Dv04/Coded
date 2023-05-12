# Write a shell script to find factorial of given number n.

read -p "Enter a number: " n

fact=1
i=1

while [ $i -le $n ]
do
    fact=`expr $fact \* $i`
    i=`expr $i + 1`
done

echo "Factorial of $n is $fact"