# This is the code for playfair cypher

inp = input("Enter the plain Text: ").lower()

matrix = [[] for i in range(5)]
alpha = "abcdefghijklmnopqrstuvwxyz0123456789"
alpha = list(alpha)


def create_frequency_dict(inp):
    inp = inp.replace(" ", "")
    inp = inp.replace(".", "")
    frequency_dict = {char: 0 for char in alpha}
    for char in inp:
        frequency_dict[char] += 1
    return frequency_dict


frequency_dict = create_frequency_dict(inp)
print(frequency_dict)


def create_matrix(inp):
    inp = inp.replace(" ", "")
    inp = inp.replace(".", "")
    inp = list(inp)
    inp = list(dict.fromkeys(inp))
    for i in range(5):
        for j in range(5):
            if len(inp) != 0:
                if inp[0] not in matrix[i]:
                    matrix[i].append(inp[0])
                    print(inp[0])
                    alpha.remove(inp[0])
                inp.remove(inp[0])
            else:
                if alpha[0] not in matrix[i]:
                    matrix[i].append(alpha[0])
                    alpha.remove(alpha[0])

    for i in range(5):
        print(matrix[i])


create_matrix(inp)
