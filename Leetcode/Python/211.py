# Design a data structure that supports adding new words and finding if a string matches any previously added string.

# Implement the WordDictionary class:

# WordDictionary() Initializes the object.
# void addWord(word) Adds word to the data structure, it can be matched later.
# bool search(word) Returns true if there is any string in the data structure that matches word or false otherwise. word may contain dots '.' where dots can be matched with any letter.

class WordDictionary:

    def __init__(self):
        self.root = {}

    def addWord(self, word: str) -> None:
        node = self.root
        for c in word:
            if c not in node:
                node[c] = {}
            node = node[c]
        node['*'] = False

    def search(self, w: str) -> bool:
        def dfs(node, i):
            if node == False:
                return False
            if i == L:
                return '*' in node
            if w[i] != '.':
                if w[i] not in node:
                    return False
                return dfs(node[w[i]], i+1)
            for j in node.values():
                if dfs(j, i+1):
                    return True
            return False
        node, L = self.root, len(w)
        return dfs(node, 0)


wordDictionary = WordDictionary()
wordDictionary.addWord("bad")
wordDictionary.addWord("dad")
wordDictionary.addWord("mad")
print(wordDictionary.search("pad"))  # return False
print(wordDictionary.search("bad"))  # return True
print(wordDictionary.search("b.."))  # return True
print(wordDictionary.search(".ad"))  # return True
