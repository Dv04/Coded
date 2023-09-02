// You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.

// You may assume the two numbers do not contain any leading zero, except the number 0 itself.

#include <stdio.h>
#include <stdlib.h>

struct ListNode
{
    int val;
    struct ListNode *next;
};

/*
    Logic:
        Take values from both the lists.
        Assign them to pointers l1 and l2.
        define a structure l3.
        define a pointer p to l3.
        define a variable carry to store the carry.
        run a while loop until both the lists are empty.
        
*/

struct ListNode *addTwoNumbers(struct ListNode *l1, struct ListNode *l2)
{
    struct ListNode *head = (struct ListNode *)malloc(sizeof(struct ListNode));
    struct ListNode *p = head;
    int carry = 0;
    while (l1 != NULL || l2 != NULL)
    {
        int x = (l1 != NULL) ? l1->val : 0;
        int y = (l2 != NULL) ? l2->val : 0;
        int sum = x + y + carry;
        carry = sum / 10;
        p->next = (struct ListNode *)malloc(sizeof(struct ListNode));
        p = p->next;
        p->val = sum % 10;
        if (l1 != NULL)
        {
            l1 = l1->next;
        }
        if (l2 != NULL)
        {
            l2 = l2->next;
        }
    }
    if (carry > 0)
    {
        p->next = (struct ListNode *)malloc(sizeof(struct ListNode));
        p = p->next;
        p->val = carry;
    }
    p->next = NULL;
    return head->next;
}

void print(struct ListNode *head)
{
    struct ListNode *ptr = head;
    // printf("\nLinked List: \n{ ");
    while (ptr != NULL)
    {
        printf(" %d, %p -->", ptr->val, ptr->next);
        ptr = ptr->next;
    }
    printf(" NULL }");
}

int main()
{
    struct ListNode *l1 = (struct ListNode *)malloc(sizeof(struct ListNode));
    struct ListNode *l2 = (struct ListNode *)malloc(sizeof(struct ListNode));
    struct ListNode *p = l1;
    p->val = 8;
    p->next = (struct ListNode *)malloc(sizeof(struct ListNode));
    p = p->next;
    p->val = 4;
    p->next = (struct ListNode *)malloc(sizeof(struct ListNode));
    p = p->next;
    p->val = 3;
    p->next = NULL;
    p = l2;
    p->val = 5;
    p->next = (struct ListNode *)malloc(sizeof(struct ListNode));
    p = p->next;
    p->val = 6;
    p->next = (struct ListNode *)malloc(sizeof(struct ListNode));
    p = p->next;
    p->val = 4;
    p->next = NULL;
    struct ListNode *l3 = addTwoNumbers(l1, l2);

    struct ListNode *ptr = l3;

    printf("{ ");
    while (ptr != NULL)
    {
        printf(" %d, %p -->", ptr->val, ptr->next);
        ptr = ptr->next;
    }

    printf(" NULL }");
    printf("\n");
    return 0;
}