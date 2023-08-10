'''Input
Only one line containing two numbers A
 and B
 (1≤A,B≤103)
Output
Print 3 lines that contain the following in the same order:

"floor A
 / B
 = Floor result" without quotes.
"ceil A
 / B
 = Ceil result" without quotes.
"round A / B = Round result" without quotes.'''

a,b = input().split()
print("floor {} / {} = {}".format(a,b,int(a)//int(b)))
print("ceil {} / {} = {}".format(a,b,(int(a)//int(b) + 1)))
print(int(str(int(a)/int(b))[-1]))
if int(str(int(a)/int(b))[-1])<5:
    print("round {} / {} = {}".format(a,b,round(int(a)/int(b))))
else:
    print("round {} / {} = {}".format(a,b,round(int(a)/int(b))+1))