# You are given the root of a binary tree containing digits from 0 to 9 only.

# Each root-to-leaf path in the tree represents a number.

# For example, the root-to-leaf path 1 -> 2 -> 3 represents the number 123.
# Return the total sum of all root-to-leaf numbers. Test cases are generated so that the answer will fit in a 32-bit integer.

# A leaf node is a node with no children.

# Definition for a binary tree node.

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

from typing import Optional


class Solution:
    sum = 0
    def summation(self,root: Optional[TreeNode],sum):
        if root is None:
            return 0
        sum = sum*10 + root.val
        if root.left is None and root.right is None:
            return sum
        return self.summation(root.left,sum) + self.summation(root.right,sum)
    def sumNumbers(self, root: Optional[TreeNode]) -> int:
        
        return self.summation(root,0)
    
