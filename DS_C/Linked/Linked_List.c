/**
 * @file Linked_List.c
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief Different types of linked list
 * @version 1.0
 * @date 2022-10-17
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <stdio.h>
#include <stdlib.h>

struct Linked_List
{
    int data;
    int key;
    struct Linked_List *next;
};

void print(struct Linked_List *head)
{
    struct Linked_List *ptr = head;
    printf("Linked List: \n{ ");
    while (ptr != NULL)
    {
        printf(" %d, %p -->", ptr->data, ptr->next);
        ptr = ptr->next;
    }
    printf(" NULL }");
}

int isEmpty(struct Linked_List *head)
{
    if (head == NULL)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

void insert_at_beginning(struct Linked_List **head, int data)
{
    struct Linked_List *ptr = (struct Linked_List *)malloc(sizeof(struct Linked_List));
    ptr->data = data;
    ptr->next = *head;
    *head = ptr;
}

void insert_at_end(struct Linked_List **head, int data)
{
    struct Linked_List *ptr = (struct Linked_List *)malloc(sizeof(struct Linked_List));
    ptr->data = data;
    ptr->next = NULL;
    struct Linked_List *p = *head;
    while (p->next != NULL)
    {
        p = p->next;
    }
    p->next = ptr;
}

void insert_at_index(struct Linked_List **head, int data, int index)
{
    struct Linked_List *ptr = (struct Linked_List *)malloc(sizeof(struct Linked_List));
    ptr->data = data;
    struct Linked_List *p = *head;
    int i = 0;
    while (i != index - 1)
    {
        p = p->next;
        i++;
    }
    ptr->next = p->next;
    p->next = ptr;
}

void delete_at_beginning(struct Linked_List **head)
{
    isEmpty(*head);
    struct Linked_List *ptr = *head;
    *head = ptr->next;
    free(ptr);
}

void delete_at_end(struct Linked_List **head)
{
    if (isEmpty(*head))
    {
        printf("Linked List is empty");
    }
    else
    {
        struct Linked_List *p = *head;
        struct Linked_List *q = p->next;
        while (q->next != NULL)
        {
            p = p->next;
            q = q->next;
        }
        p->next = NULL;
        free(q);
    }
}

void delete_at_index(struct Linked_List **head, int index)
{
    if (!isEmpty(*head))
    {
        struct Linked_List *p = *head;
        struct Linked_List *q = p->next;
        int i = 0;
        while (i != index - 1)
        {
            p = p->next;
            q = q->next;
            i++;
        }
        p->next = q->next;
        free(q);
    }
    else
    {
        printf("Linked List is Empty");
    }
}

int main()
{
    struct Linked_List *head;
    struct Linked_List *second;
    struct Linked_List *third;

    head = (struct Linked_List *)malloc(sizeof(struct Linked_List));
    second = (struct Linked_List *)malloc(sizeof(struct Linked_List));
    third = (struct Linked_List *)malloc(sizeof(struct Linked_List));

    head->data = 1;
    head->next = second;
    second->data = 2;
    second->next = third;
    third->data = 3;
    third->next = NULL;

    printf("Press 1 for printing the linked list, 2 for Inserting in the list, 3 for deleting from the list, 4 for exit.");
    int choice;
    scanf("%d", &choice);
    switch (choice)
    {
    case 1:

        print(head);
        break;

    case 2:

        printf("Press 0 to insert at beginning, 1 to insert at end, 2 to insert at index.");
        int pos;
        scanf("%d", &pos);

        printf("Enter the element you want to insert: ");
        int data;
        scanf("%d", &data);

        switch (pos)
        {

        case 0:
            insert_at_beginning(&head, data);
            break;

        case 1:
            insert_at_end(&head, data);
            break;

        case 2:
            printf("Enter the index: ");
            int index;
            scanf("%d", &index);
            insert_at_index(&head, data, index);
            break;

        default:
            printf("Invalid position");
            break;
        }
        break;

    case 3:

        printf("Press 0 to delete at beginning, 1 to delete at end, 2 to delete at index.");
        int pos1;
        scanf("%d", &pos1);

        switch (pos1)
        {

        case 0:
            delete_at_beginning(&head);
            break;

        case 1:
            delete_at_end(&head);
            break;

        case 2:
            printf("Enter the index: ");
            int index;
            scanf("%d", &index);
            delete_at_index(&head, index);
            break;

        default:
            printf("Invalid position");
            break;
        }
        break;

    case 4:
        exit(0);
        break;

    default:
        printf("Invalid choice");
        break;
    }

    printf("\n");

    return 0;
}