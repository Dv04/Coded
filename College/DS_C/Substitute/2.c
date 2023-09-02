/**
 * @file 1.c
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief WAP in C to delete a node from the Middle of Singly Linked List
 * @version 1.0
 * @date 2022-12-12
 *
 * @copyright Copyright (c) 2022
 *
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

void insert_at_index(struct Node **head, int data, int index)
{
    struct Node *ptr = (struct Node *)malloc(sizeof(struct Node));
    ptr->data = data;
    struct Node *p = *head;
    int i = 0;
    while (i != index - 1)
    {
        p = p->next;
        i++;
    }
    ptr->next = p->next;
    p->next = ptr;
}

int main()
{
    struct Node *second;
    struct Node *third;
    struct Node *four;
    int data;
    int n = 4;

    head = (struct Node *)malloc(sizeof(struct Node));
    second = (struct Node *)malloc(sizeof(struct Node));
    third = (struct Node *)malloc(sizeof(struct Node));
    four = (struct Node *)malloc(sizeof(struct Node));



    head->data = 1;
    head->next = second;
    second->data = 2;
    second->next = third;
    third->data = 3;
    third->next = four;
    four->data = 4;
    four->next = NULL;

    printf("Code for deleting the middle element");

    print(head);

    printf("\n\nWhich Data do you want to Insert: ");
    scanf("%d", &data);

    insert_at_index(&head, data, (int)(n / 2));

    print(head);

    printf("\n");

    return 0;
}