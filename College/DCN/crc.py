def xor(a, b):
    result = []
    for i in range(1, len(b)):
        if a[i] == b[i]:
            result.append('0')
        else:
            result.append('1')
    return ''.join(result)

def mod2div(dividend, divisor):
    pick = len(divisor)
    tmp = dividend[0: pick]
    while pick < len(dividend):
        if tmp[0] == '1':
            tmp = xor(divisor, tmp) + dividend[pick]
        else:
            tmp = xor('0'*pick, tmp) + dividend[pick]
        pick += 1
    if tmp[0] == '1':
        tmp = xor(divisor, tmp)
    else:
        tmp = xor('0'*pick, tmp)
    checkword = tmp
    return checkword

def encodeData(data, key):
    l_key = len(key)
    appended_data = data + '0'*(l_key-1)
    remainder = mod2div(appended_data, key)
    codeword = data + remainder
    print("Remainder : ", remainder)
    print("Encoded Data (Data + Remainder) : ", codeword)
    print("\nReceved the Data: ",codeword)
    def decodeData(codeword, key):
        l_key = len(key)
        remainder = mod2div(codeword, key)
        if '1' in remainder:
            print("Error detected in transmission.")
            # Perform error correction if desired
        else:
            print("No errors detected in transmission.")
        data = codeword[:-l_key]
        return data
    data = decodeData(codeword, key)
    print("\nOriginal Data Sent : ", data)

data = str(input("Enter the data: "))
key = str(input("Enter the key: "))
encodeData(data, key)