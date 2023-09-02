# stuffing the stuffed signal

def stuff(sig):
    onecounter = 0  
    index = 0  
    one = []  # one indexes
    signal = list(sig)
    for i in signal:
        index += 1
        if i == '0':
            onecounter = 0
        else:
            onecounter += 1
        if onecounter == 5:
            one.append(index)
            onecounter = 0
    k = 0  # count extra index number
    for i in one:
        # print(i)
        signal.insert(i + k, '0')
        k += 1
    return signal


# destuffing the stuffed signal
def destuff(sig):
    onecounter = 0  
    index = 0   
    one = [] 
    sig = list(sig)
    for i in sig:
        index += 1
        if i == '0':
            onecounter = 0
        else:
            onecounter += 1
        if onecounter == 5:
            one.append(index)
            onecounter = 0
    k = 0  # count extra index number
    for i in one:
        # print(i)
        sig.pop(i + k)
        k -= 1
    return sig

signal = input("Enter the signal: ")

for i in signal:
    if i != '0' and i != '1':
        print("Invalid Signal")
        exit()
        
print("Original Signal : ", signal)

stuffed = stuff(signal)
print("Stuffed Signal: ", end="")
print("".join([a for a in stuffed]))

destuffed = destuff(stuffed)
print("Destuffed Signal: ", end="")
print("".join([a for a in destuffed]))