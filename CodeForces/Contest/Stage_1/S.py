a = float(input())
if a<0 or a>100:
    print("Out of Intervals")
elif a <= 25:
    print("Interval [0,25]")
elif a<=50:
    print("Interval (25,50]")
elif a<=75:
    print("Interval (50,75]")
else:
    print("Interval (75,100]")