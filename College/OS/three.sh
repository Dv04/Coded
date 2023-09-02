# Write a shell script which will accept a number n and display first n prime numbers as output.

read -p "Enter the number of prime numbers to be displayed: " n

count=0
num=2

while [ $count -lt $n ]
do
    i=2
    flag=0
    while [ $i -lt $num ]
    do
        if [ `expr $num % $i` -eq 0 ]
        then
            flag=1
            break
        fi
        i=`expr $i + 1`
    done
    if [ $flag -eq 0 ]
    then
        echo $num
        count=`expr $count + 1`
    fi
    num=`expr $num + 1`
done
