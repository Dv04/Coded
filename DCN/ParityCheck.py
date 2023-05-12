# A code to check for Odd and Even parity. Also print error if not parity and print the code if parity

n = input("Enter the code to be checked: ")
count = 0
pari = -1

def EvenParSend(n, count, pari):
    print("\nEven Parity Checker Initialised for sending")
    for i in n:
        if i == '1':
            count = count + 1
            
    if count%2 == 0:
        print("Parity bit is 0\n")
        pari = '0'
        
    else:
        print("Parity bit is 1\n")
        pari = '1'
        
    n += pari
    print("The code to be sent is: ", n)
    EvenParRec(n, count, pari)

def EvenParRec(n, count, pari):

    print("\nEven Parity Checker Initialised for receiving")
    print("The code received is: ", n)
    count = 0
    for i in n:
        if i == '1':
            count = count + 1
            
    print("Number of 1's in the code is: ", count)
        
    if count%2 == 0:
        print("No error in the code\n")
        
    else:
        print("Error in the code\n")

def OddParSend(n, count, pari):
    print("\nOdd Parity Checker Initialised for sending")
    for i in n:
        if i == '1':
            count = count + 1
            
    if count%2 == 0:
        print("Parity bit is 1\n")
        pari = '1'
        
    else:
        print("Parity bit is 0\n")
        pari = '0'
        
    n += pari
    print("The code to be sent is: ", n)
    OddParRec(n, count, pari)

def OddParRec(n, count, pari):
    print("\nOdd Parity Checker Initialised for receiving")
    print("The code received is: ", n)
    count = 0
    for i in n:
        if i == '1':
            count = count + 1
            
    print("Number of 1's in the code is: ", count)
        
    if count%2 == 0:
        print("Error in the code\n")
        
    else:
        print("No error in the code\n")

print("\nEnter 1 for Even Parity Check")
print("Enter 2 for Odd Parity Check")

choice = int(input("\nEnter your choice: "))

if choice == 1:
    EvenParSend(n, count, pari)    
elif choice == 2:
    OddParSend(n, count, pari)    
else:
    print("Wrong Choice")