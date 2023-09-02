In = input("PLease enter the character you want to print : ")
h = int(input("Enter the number of lines to print : "))

for x in range(h):
    print(" " * (h - x), In * (x + 1))
for x in range(h-2, -1, -1):
    print(" " * (h - x), In * (x + 1))