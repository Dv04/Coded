a = int(input("Enter a number: "))
for i in range(1,int(a**0.5)+1):
    if a%i==0:
        print(i,sep = " ")
        print(a//i,sep = " ")
        