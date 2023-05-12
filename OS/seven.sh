# Write a shell script to display all executable files, directories and zero sized files from current directory

for file in *
do
    if [ -f $file ]
    then
        if [ -x $file ]
        then
            echo "$file is an executable file"
        fi
        if [ ! -s $file ]
        then
            echo "$file is a zero sized file"
        fi
    elif [ -d $file ]
    then
        echo "$file is a directory"
    fi
done