# Checksum Code

n = input("Enter the code to be sent: ")
k = input("Enter the block size: ")

n+="0"*(3-(len(n)%int(k)))
arr = []
for i in range(0, len(n), int(k)):
    arr.append(n[i:i+int(k)])

print(arr)