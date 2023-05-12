/**
 * @file Node.c
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief WAP to implement following operations on singly linked list:
 *        1. Insert a node at the front of the linked list
 *        2. Delete the first node of the linked list
 *        3. Insert a node at the end of the linked list
 *        4. Delete a node after specified position
 *        5. Delete a node before specified position.
 *        6. Insert a node such that linked list is in ascending order
 *
 * @version 1.0
 * @date 2022-10-17
 *
 * @copyright Copyright (c) 2022
 *
 */

/*

    The Logic:
        1. Create a structure of linked list.
        2. Insert three nodes at the front of the linked list.
        3. Ask user for choice of direction
        4. If user wants to insert a node at the front of the linked list then ask user for data and call insert_at_beginning function.
            4.1 Make a new structure of linked list.
            4.2 Insert the new node at the front of the linked list.

        5. If user wants to delete the first node of the linked list then call delete_first_node function.
            5.1 Check if the linked list is empty or not.
            5.2 If yes then print the message.
            5.3 If no then delete the first node of the linked list.

        6. If the user want to insert a node at the end of the linked list then ask user for data and call insert_at_end function.
            6.1 Make a new structure of linked list.
            6.2 Insert the new node at the end of the linked list.

        7. If the user want to delete a node after specified position then ask user for position and call delete_at_index function.
            7.1 Check if the linked list is empty or not.
            7.2 If yes then print the message.
            7.3 If no then delete the node after specified position.

        8. If the user want to delete a node before specified position then ask user for position and call delete_at_index function.
            8.1 Check if the linked list is empty or not.
            8.2 If yes then print the message.
            8.3 If no then delete the node before specified position.

        9. If the user want to insert a node such that linked list is in ascending order then ask user for data and call insert_in_ascending_order function.
            9.1 Make a new structure of linked list.
            9.2 Insert the new node such that linked list is in ascending order.





*/

#include <stdio.h>
#include <stdlib.h>

struct Node
{
    int data;
    struct Node *next;
};

struct Node *head, *tail = NULL;

int isEmpty(struct Node *head)
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

void print(struct Node *head)
{
    struct Node *ptr = head;
    printf("\nLinked List: \n{ ");
    while (ptr != NULL)
    {
        printf(" %d, %p -->", ptr->data, ptr->next);
        ptr = ptr->next;
    }
    printf(" NULL }");
}

void insert_at_beginning(struct Node **head, int data)
{
    struct Node *ptr = (struct Node *)malloc(sizeof(struct Node));
    ptr->data = data;
    ptr->next = *head;
    *head = ptr;
}

void insert_at_end(struct Node **head, int data)
{
    struct Node *ptr = (struct Node *)malloc(sizeof(struct Node));
    ptr->data = data;
    ptr->next = NULL;
    struct Node *p = *head;
    while (p->next != NULL)
    {
        p = p->next;
    }
    p->next = ptr;
}

void delete_at_beginning(struct Node **head)
{
    if (isEmpty(*head))
    {
        printf("\nLinked List is currently empty.\n");
    }
    else
    {
        printf("\nDeleting %d from the beginning of the Linked List.\n", (*head)->data);
        struct Node *ptr = *head;
        *head = (*head)->next;
        free(ptr);
        // delete_at_beginning(head);
    }
}

void delete_at_index(struct Node **head, int index)
{
    if (index == 1)
    {
        delete_at_beginning(head);
    }
    if (!isEmpty(*head))
    {
        printf("\nPress\n\n 1 for at the index, 2 for before the index, 3 for after the index\n\nWhat is your choice: ");
        int choice;
        scanf("%d", &choice);
        switch (choice)
        {
        case 1:
            printf("\nDeleting %d from the index %d of the Linked List.\n", (*head)->data, index);
            struct Node *p = *head;
            struct Node *q = p->next;
            int i = 0;
            while (i != index - 1)
            {
                p = p->next;
                q = q->next;
                i++;
            }
            p->next = q->next;
            free(q);
            break;

        case 2:
            printf("\nDeleting %d from the index %d of the Linked List.\n", (*head)->data, index - 1);
            struct Node *r = *head;
            struct Node *s = r->next;
            int j = 0;
            while (j != index - 2)
            {
                r = r->next;
                s = s->next;
                j++;
            }
            r->next = s->next;
            free(s);
            break;

        case 3:
            printf("\nDeleting %d from the index %d of the Linked List.\n", (*head)->data, index + 1);
            struct Node *t = *head;
            struct Node *u = t->next;
            int k = 0;
            while (k != index)
            {
                t = t->next;
                u = u->next;
                k++;
            }
            t->next = u->next;
            free(u);
            break;
        }
    }
    else
    {
        printf("Linked List is Empty");
    }
}

int search(struct Node *head, int data)
{
    struct Node *ptr = head;
    int index = 1;
    while (ptr != NULL)
    {
        if (ptr->data == data)
        {
            printf("\nLinked List contains %d at index %d.\n", data, index);
            return 1;
        }
        ptr = ptr->next;
        index++;
        if (ptr == NULL)
        {
            printf("\nLinked List does not contain %d.\n", data);
            return 0;
        }
    }
    return 0;
}

void sortList()
{
    // Node current will point to head
    struct Node *current = head, *index = NULL;
    int temp;

    if (head == NULL)
    {
        return;
    }
    else
    {
        while (current != NULL)
        {
            // Node index will point to Node next to current
            index = current->next;

            while (index != NULL)
            {
                // If current Node's data is greater than index's Node data, swap the data between them
                if (current->data > index->data)
                {
                    temp = current->data;
                    current->data = index->data;
                    index->data = temp;
                }
                index = index->next;
            }
            current = current->next;
        }
    }
}

int insert_in_ascending_order(struct Node **head, int data)
{
    struct Node *ptr = (struct Node *)malloc(sizeof(struct Node));
    ptr->data = data;
    ptr->next = NULL;
    struct Node *p = *head;
    if (p->data > data)
    {
        ptr->next = p;
        *head = ptr;
        return 1;
    }
    while (p->next != NULL)
    {
        if (p->next->data > data)
        {
            ptr->next = p->next;
            p->next = ptr;
            return 1;
        }
        p = p->next;
    }
    p->next = ptr;
    return 1;
}

int main()
{
    struct Node *second;
    struct Node *third;

    head = (struct Node *)malloc(sizeof(struct Node));
    second = (struct Node *)malloc(sizeof(struct Node));
    third = (struct Node *)malloc(sizeof(struct Node));

    head->data = 1;
    head->next = second;
    second->data = 2;
    second->next = third;
    third->data = 3;
    third->next = NULL;
    printf("\n\nPress:\n\n 1 for printing the linked list\n 2 for Inserting in the list\n 3 for Deleting the whole list\n 4 for searching a particular data\n 5 for exit\n 6 for sorting\n 7 for insertion in sorted");
    while (1)
    {
        printf("\n\nWhat is your choice: ");
        int choice;
        scanf("%d", &choice);
        switch (choice)
        {
        case 1:

            print(head);
            break;

        case 2:
            printf("\nPress:\n\n 1 for Inserting at the beginning\n 2 for Inserting at the end\n\nEnter your choice: ");
            int choice2;
            scanf("%d", &choice2);
            switch (choice2)
            {
            case 1:
                printf("\nEnter the data to be inserted at the beginning: ");
                int data;
                scanf("%d", &data);
                insert_at_beginning(&head, data);
                break;

            case 2:
                printf("\nEnter the data to be inserted at the end: ");
                int data2;
                scanf("%d", &data2);
                insert_at_end(&head, data2);
                break;

            default:
                printf("\nInvalid Choice.\n");
                break;
            }
            break;

        case 3:
            printf("\nPress:\n\n 1 for deleting at beginning\n 2 for deleting with given index\n\nEnter your choice: ");
            int choice;
            scanf("%d", &choice);
            switch (choice)
            {
            case 1:
                delete_at_beginning(&head);
                print(head);
                break;

            case 2:
                printf("Enter the index: ");
                int index;
                scanf("%d", &index);
                delete_at_index(&head, index);
                print(head);
                break;

            default:
                printf("Invalid choice");
                break;
            }
            break;

        case 4:
            printf("Enter the data to be searched: ");
            int data1;
            scanf("%d", &data1);
            search(head, data1);
            break;

        case 5:
            exit(0);
            break;

        case 6:
            sortList();
            break;

        case 7:
            printf("\nEnter the data to be inserted: ");
            int data;
            scanf("%d", &data);
            insert_in_ascending_order(&head, data);

        default:
            printf("Invalid choice");
            break;
        }
    }
    printf("\n");

    return 0;
}