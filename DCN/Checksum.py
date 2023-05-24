# Checksum Code

n = input("\n\nEnter the code to be sent: ")
k = input("Enter the block size: ")

def ReceiveCheck(ans, k):
    print("\n\nEntering the Receiving Mode with the Code: ",ans)
    
    arr1 = []
    for i in range(0, len(ans), int(k)):
        arr1.append(ans[i:i+int(k)])
    
    result = "0" * int(k)
    for block in arr1:
        carry = 0
        for i in range(int(k)):
            temp = int(result[i]) + int(block[i]) + carry
            result = result[:i] + str(temp % 2) + result[i+1:]
            carry = temp // 2

    if carry != 0:
        result = str(carry) + result[1:]
        
    print("Summation of the Received code with block size ",k," is: ",result)
    print("\n------------------------------------------------------------\n")
    print("As the summation is not zero, there is no error in the code.\n")
    
    

def SentCheck(n, k):
    print("\n------------------------------------------------------------\n")    
    ans = "0"*(int(k)-(len(n)%int(k)))
    ans+=n
    print("Entering the Sending Mode with the code: ", ans)
    print("\n------------------------------------------------------------\n")

    print("The code is divided in to parts of size ",k," as follows: ")
    arr = []
    for i in range(0, len(ans), int(k)):
        print(ans[i:i+int(k)])
        arr.append(ans[i:i+int(k)])

    print("\n------------------------------------------------------------")
    result = "0" * int(k)
    for block in arr:
        carry = 0
        for i in range(int(k)):
            temp = int(result[i]) + int(block[i]) + carry
            result = result[:i] + str(temp % 2) + result[i+1:]
            carry = temp // 2

    if carry != 0:
        result = str(carry) + result[1:]

    print("\nSummation of binary blocks (with carry):", result)
    
    ans1 = ""
    for i in range(len(result)):
        if result[i] == '0':
            ans1 += '1'
        else:
            ans1 += '0'
            
    print("Checksum of the entered code is: ", ans1)
    print("Sending the code with the checksum: ", ans+ans1,"")
    
    ReceiveCheck(ans+ans1,k)

SentCheck(n, k)