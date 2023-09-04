# Count Frequency of each element in an array.

def freq_count(array, freq):
    for start in array:
        if start not in freq:
            freq[start] = 0
        freq[start] += 1

    
array = [a for a in map(int,input("Enter the array: ").split())]
freq = {}
freq_count(array, freq)
print(freq)

# Print the highest and lowest frequency element.

print(min(freq, key = freq.get))
print(max(freq, key = freq.get))