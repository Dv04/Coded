# Take input of all the elements as a string 
s = input()

# Use map function to wrap-up them and converting to desired data type.
li = list (map (int, s.split()))

print("{} {}".format(min(li),max(li)))