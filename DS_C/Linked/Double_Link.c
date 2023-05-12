/**
 * @file Double_Link.c
 * @author Dev Sanghvi (dev04san@gmail.com)
 * @brief Double Linkedlist
 * @version 1.0
 * @date 2022-11-28
 *
 * @copyright Copyright (c) 2022
 *
 */

/*  Write a program to perform following operations on a circular linked list:
        a) Insert a node at the beginning of the linked list.
        b) Insert a node at the end of the linked list.
        c) Delete first node of the linked list.
        d) Delete last node of the circular linked list.
 */

#include "stdio.h"
#include "stdlib.h"

struct node
{
    int data;
    struct node *next;
    struct node *prev;
};

struct node *head = NULL;

void display()
{
    struct node *temp;
    temp = head;
    if (head == NULL)
    {
        printf("List is empty.\n");
    }
    else
    {
        printf("List is: ");
        while (temp->next != head)
        {
            printf("%d ", temp->data);
            temp = temp->next;
        }
        printf("%d ", temp->data);
        printf("\nList Printed.\n");
    }
}

void insert_beg()
{
    struct node *new_node;
    new_node = (struct node *)malloc(sizeof(struct node));
    printf("Enter the data: ");
    scanf("%d", &new_node->data);
    new_node->next = NULL;
    new_node->prev = NULL;
    if (head == NULL)
    {
        head = new_node;
        new_node->next = head;
        new_node->prev = head;
    }
    else
    {
        struct node *temp;
        temp = head;
        while (temp->next != head)
        {
            temp = temp->next;
        }
        temp->next = new_node;
        new_node->prev = temp;
        new_node->next = head;
        head->prev = new_node;
        head = new_node;
    }
    printf("Node inserted at the beginning.\n");
}

void insert_end()
{
    struct node *new_node;
    new_node = (struct node *)malloc(sizeof(struct node));
    printf("Enter the data: ");
    scanf("%d", &new_node->data);
    new_node->next = NULL;
    new_node->prev = NULL;
    if (head == NULL)
    {
        head = new_node;
        new_node->next = head;
        new_node->prev = head;
    }
    else
    {
        struct node *temp;
        temp = head;
        while (temp->next != head)
        {
            temp = temp->next;
        }
        temp->next = new_node;
        new_node->prev = temp;
        new_node->next = head;
        head->prev = new_node;
    }
    printf("Node inserted at the end.\n");
}

void delete_beg()
{
    if (head == NULL)
    {
        printf("List is empty.\n");
    }
    else
    {
        struct node *temp;
        temp = head;
        while (temp->next != head)
        {
            temp = temp->next;
        }
        temp->next = head->next;
        head->next->prev = temp;
        free(head);
        head = temp->next;
        printf("Node deleted from the beginning.\n");
    }
}

void delete_end()
{
    if (head == NULL)
    {
        printf("List is empty.\n");
    }
    else
    {
        struct node *temp;
        temp = head;
        while (temp->next != head)
        {
            temp = temp->next;
        }
        temp->prev->next = head;
        head->prev = temp->prev;
        free(temp);
        printf("Node deleted from the end.\n");
    }
}

int main()
{
    int choice;
    printf("\n\nEnter 1 to display the list.\nEnter 2 to insert a node at the beginning.\nEnter 3 to insert a node at the end.\nEnter 4 to delete a node from the beginning.\nEnter 5 to delete a node from the end.\nEnter 6 to exit ");
    while (1)
    {
        printf("\nEnter your choice: ");
        scanf("%d", &choice);
        switch (choice)
        {
        case 1:
            display();
            break;
        case 2:
            insert_beg();
            break;
        case 3:
            insert_end();
            break;
        case 4:
            delete_beg();
            break;
        case 5:
            delete_end();
            break;
        case 6:
            exit(0);
            break;
        default:
            printf("Invalid choice.\n");
            break;
        }
    }
    return 0;
}
