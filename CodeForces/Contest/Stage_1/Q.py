a,b = map(float,input().split())
if a<0:
    if b<0:
        print("Q3")
    elif b>0:
        print("Q2")
    else:
        print("Eixo X")
    
elif a>0:
    if b<0:
        print("Q4")
    elif b>0:
        print("Q1")
    else:
        print("Eixo X")
else:
    if b==0:
        print("Origem")
    else:
        print("Eixo Y")