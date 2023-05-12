# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def hasCycle(self, head: optional[listNode]) -> bool:
        ans = []
        while (head):
            print(head.data)
            ans.append(head.val)
            head = head.next

  