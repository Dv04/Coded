t = int(input())



def SubArr(arr, start, end):
    if end == len(arr):
        return
    elif start > end:
        return SubArr(arr, 0, end + 1)
    else:
        # print(arr[start:end + 1])
        arra.append(arr[start:end + 1])
        arra[-1].sort()
        mex = 0
        for idx in range(len(arra[-1])):
            if arra[-1][idx] == mex:

                # Increment mex
                mex += 1
        mex1.append(mex)
        return SubArr(arr, start + 1, end)


for i in range(t):
    arra = []
    mex1 = []
    n = list(map(int, input().split()))
    xnum = list(map(int, input().split()))
    SubArr(xnum, 0, 0)
    a = {tuple(k):v for k, v in zip(arra, mex1)}
    for j in range(n[1]):
        count = 0
        b = list(map(int, input().split()))
        for k in a:
            if len(k) in range(b[0],b[1]+1) and a[k] in range(b[2],b[3]+1):
                count+=1
        print(count)
    