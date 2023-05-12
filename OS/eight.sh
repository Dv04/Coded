# Write a shell script to check entered string is palindrome or not.

read -p "Enter a string: " str

len=${#str}
i=0
j=`expr $len - 1`
flag=0
while [ $i -lt $j ]
do
    if [ ${str:$i:1} != ${str:$j:1} ]
    then
        flag=1
        break
    fi
    i=`expr $i + 1`
    j=`expr $j - 1`
done

if [ $flag -eq 0 ]
then
    echo "$str is a palindrome"
else
    echo "$str is not a palindrome"
fi

