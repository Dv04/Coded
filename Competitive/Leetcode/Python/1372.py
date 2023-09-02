# You are given the root of a binary tree.

# A ZigZag path for a binary tree is defined as follow:

# Choose any node in the binary tree and a direction (right or left).
# If the current direction is right, move to the right child of the current node; otherwise, move to the left child.
# Change the direction from right to left or from left to right.
# Repeat the second and third steps until you can't move in the tree.
# Zigzag length is defined as the number of nodes visited - 1. (A single node has a length of 0).

# Return the longest ZigZag path contained in that tree.

# Definition for a binary tree node.

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

from pyparsing import Optional


class Solution:
    
    def longestZigZag(self, root: Optional[TreeNode]) -> int:
        
        count = 0
        
        def dfs(node, direction):
            if not node:
                return 0
            if direction == 'left':
                return 1 + dfs(node.left, 'right')
            else:
                return 1 + dfs(node.right, 'left')
        
        if root:
            count = max(count, dfs(root.left, 'right'), dfs(root.right, 'left'))
            count = max(count, self.longestZigZag(root.left), self.longestZigZag(root.right))
            
        return count
    
print(Solution().longestZigZag(root = [1,null,1,1,1,null,null,1,1,null,1,null,null,null,1,null,1]))