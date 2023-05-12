/**
 * @file Circular_Link.c
 * @author Dev Sanghvi (dev04san@gmail.com)
 * @brief Circular Linkedlist
 * @version 1.0
 * @date 2022-11-28
 *
 * @copyright Copyright (c) 2022
 *
 */

/**
 * Write a program to perform following operations on a circular linked list:
        a) Insert a node at the beginning of the linked list.
        b) Insert a node at the end of the linked list.
        c) Delete first node of the linked list.
        d) Delete last node of the circular linked list.
 */

/*
    Logic:
        1. Create a structure of linked list.
        2. Ask user for choice.
        3. If user wants to insert at beginning.
            3.1 Ask user for data.
            3.2 Create a new node.
            3.3 If head is NULL, then make new node as head.
            3.4 Else, traverse till last node and make new node as head.
        4. If user wants to insert at end.
            4.1 Ask user for data.
            4.2 Create a new node.
            4.3 If head is NULL, then make new node as head.
            4.4 Else, traverse till last node and insert there.

*/

#include <stdio.h>
#include <stdlib.h>

struct node
{
    int data;
    struct node *next;
};

struct node *head = NULL;

void display()
{
    struct node *ptr;
    ptr = head;
    if (head == NULL)
    {
        printf("\nList is empty.\n");
    }
    else
    {
        printf("\nElements are: \n");
        do
        {
            printf("%d ", ptr->data);
            ptr = ptr->next;
        } while (ptr != head);
        printf("\n\nList printed. \n");
    }
}

void beginsert()
{
    struct node *ptr, *temp;
    int item;
    ptr = (struct node *)malloc(sizeof(struct node));
    if (ptr == NULL)
    {
        printf("Overflow");
    }
    else
    {
        printf("\nEnter the value: ");
        scanf("%d", &item);
        ptr->data = item;
        if (head == NULL)
        {
            head = ptr;
            ptr->next = head;
        }
        else
        {
            temp = head;
            while (temp->next != head)
            {
                temp = temp->next;
            }
            temp->next = ptr;
            ptr->next = head;
            head = ptr;
        }
    }
    display();
}

void endinsert()
{
    struct node *ptr, *temp;
    int item;
    ptr = (struct node *)malloc(sizeof(struct node));
    if (ptr == NULL)
    {
        printf("Overflow");
    }
    else
    {
        printf("\nEnter the value: ");
        scanf("%d", &item);
        ptr->data = item;
        if (head == NULL)
        {
            head = ptr;
            ptr->next = head;
        }
        else
        {
            temp = head;
            while (temp->next != head)
            {
                temp = temp->next;
            }
            temp->next = ptr;
            ptr->next = head;
        }
    }
    display();
}

void begindelete()
{
    struct node *ptr;
    if (head == NULL)
    {
        printf("Underflow");
    }
    else if (head->next == head)
    {
        head = NULL;
        free(head);
        printf("Node deleted");
    }
    else
    {
        ptr = head;
        while (ptr->next != head)
        {
            ptr = ptr->next;
        }
        ptr->next = head->next;
        free(head);
        head = ptr->next;
    }
    display();
}

void enddelete()
{
    struct node *ptr, *pre;
    if (head == NULL)
    {
        printf("\nUnderflow\n");
    }
    else if (head->next == head)
    {
        head = NULL;
        free(head);
        printf("\nNode deleted.\n");
    }
    else
    {
        ptr = head;
        while (ptr->next != head)
        {
            pre = ptr;
            ptr = ptr->next;
        }
        pre->next = head;
        free(ptr);
        printf("\nNode deleted.\n");
    }
    display();
}

int main()
{
    int choice = 0;
    printf("\n\nEnter 1 to insert at beginning.\nEnter 2 to insert at end \nEnter 3 to delete at beginning \nEnter 4 to delete at the end \nEnter 5 to display.\nEnter 6 to Exit.\n");
    while (1)
    {
        printf("\nWhat is your choice: ");
        scanf("%d", &choice);
        switch (choice)
        {
        case 1:
            beginsert();
            break;
        case 2:
            endinsert();
            break;
        case 3:
            begindelete();
            break;
        case 4:
            enddelete();
            break;
        case 5:
            display();
            break;
        case 6:
            exit(0);
            break;
        default:
            printf("Invalid choice. \n");
        }
    }
    return 0;
}