# import filecmp

# seta = set()

# with open("temp.txt", "r") as f:
#     name = f.readlines()
#     for i in name:
#         seta.add(i.strip().lower())  # Ignore case while adding to set

# print(len(seta))

# with open("temp.txt", "w") as f:
#     for i in seta:
#         f.write(i + "\n")


def find_missing_names(file1, file2):
    with open(file1, "r") as f:
        names1 = set(line.strip() for line in f)
    with open(file2, "r") as f:
        names2 = set(line.strip() for line in f)

    return names1 - names2, names2 - names1


# Usage
missing_in_file1, missing_in_file2 = find_missing_names("name.txt", "temp.txt")
print("Missing in name.txt:", missing_in_file1)
print("Missing in temp.txt:", missing_in_file2)
