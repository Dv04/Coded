# Checksum Code

n = input("Enter the code to be sent: ")
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
        
    print(result)
    

def SentCheck(n, k):
    ans = "0"*(int(k)-(len(n)%int(k)))
    ans+=n
    print("\n\nEntering the Sending Mode with the code: ", ans)

    arr = []
    for i in range(0, len(ans), int(k)):
        arr.append(ans[i:i+int(k)])

    result = "0" * int(k)
    for block in arr:
        carry = 0
        for i in range(int(k)):
            temp = int(result[i]) + int(block[i]) + carry
            result = result[:i] + str(temp % 2) + result[i+1:]
            carry = temp // 2

    if carry != 0:
        result = str(carry) + result[1:]

    print("Summation of binary blocks (with carry):", result)

    print("The Code to be sent is: ",ans+result)
    ans+=result
    
    ans1 = ""
    for i in range(len(ans)):
        if ans[i] == '0':
            ans1 += '1'
        else:
            ans1 += '0'
            
    ReceiveCheck(ans1,k)

SentCheck(n, k)