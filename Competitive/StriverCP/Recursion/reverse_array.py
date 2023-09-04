# Reverse a given array using recursion.

def reverse(array, start, end):
    if start >= end:
        return
    array[start], array[end] = array[end], array[start]
    reverse(array, start+1, end-1)
    
array = [a for a in map(int,input("Enter the array: ").split())]
reverse(array, 0, len(array)-1)
print(array)
