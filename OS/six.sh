# Write a shell script to read n numbers as command arguments and sort them in descending order.

read -p "Enter the numbers: " -a arr

n=${#arr[@]}
i=0
while [ $i -lt $n ]
do
    j=`expr $i + 1`
    while [ $j -lt $n ]
    do
        if [ ${arr[$i]} -lt ${arr[$j]} ]
        then
            temp=${arr[$i]}
            arr[$i]=${arr[$j]}
            arr[$j]=$temp
        fi
        j=`expr $j + 1`
    done
    i=`expr $i + 1`
done

echo "Sorted array in descending order: ${arr[@]}"