# Given a singly linked list, return a random node's value from the linked list. Each node must have the same probability of being chosen.

# Implement the Solution class:

# Solution(ListNode head) Initializes the object with the head of the singly-linked list head.
# int getRandom() Chooses a node randomly from the list and returns its value. All the nodes of the list should be equally likely to be chosen.

from typing import Optional
import random

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:

    
    
    def __init__(self, head: Optional[ListNode]):
        self.head = head

    def getRandom(self) -> int:
        node = self.head
        res = node.val
        for i in range(1, 1000000):
            node = node.next
            if not node:
                break
            if random.randint(1, i+1) == 1:
                res = node.val
        return res
        


# Your Solution object will be instantiated and called as such:
# obj = Solution(head)
# param_1 = obj.getRandom()