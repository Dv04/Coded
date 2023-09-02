/**
 * @file Queue.c
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief Queue Applications.
 * @version 1.0
 * @date 2022-10-10
 *
 * @copyright Copyright (c) 2022
 *
 */

/*
    Write a program for queue to perform insertion and deletion of data with the help of Add and Delete operation.
*/

#include <stdio.h>

int isFull(int *front, int *rear, int size)
{
    if (*rear == size - 1)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

int isEmpty(int *front, int *rear)
{
    if (*front == 0 && *rear == -1)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

void display(int *queue, int *front, int *rear)
{
    if (isEmpty(front, rear))
    {
        printf("\nQueue is Empty. Cannot Display.\n");
    }
    else
    {
        for (int i = *front; i <= *rear; i++)
        {
            printf("%d\n", queue[i]);
        }
    }
}

void Add(int *queue, int *front, int *rear, int size, int data)
{
    if (isFull(front, rear, size))
    {
        printf("\nQueue is Full. Cannot Add.\n");
    }
    else
    {
        if (*rear == size - 1)
        {
            *rear = -1;
        }
        *rear = *rear + 1;
        queue[*rear] = data;
    }

    display(queue, front, rear);
}

int Delete(int *queue, int *front, int *rear)
{
    int data;
    if (isEmpty(front, rear))
    {
        printf("\nQueue is Empty. Cannot Delete.\n");
    }
    else
    {
        data = queue[*front];
        if (*front == *rear)
        {
            *front = 0;
            *rear = -1;
        }
        else
        {
            *front = *front + 1;
        }
        display(queue, front, rear);
    }

    return data;
}

int main()
{

    int front = 0, rear = -1, size = 5;
    int queue[size], data;
    int choice;
    printf("\n1. Addition\n2. Deletion\n3. Display\n4. Exit\n");

    do
    {
        printf("\nEnter your choice: ");
        scanf("%d", &choice);
        switch (choice)
        {
        case 1:
            printf("\nEnter the data to be Add: ");
            scanf("%d", &data);
            Add(queue, &front, &rear, size, data);
            break;
        case 2:
            data = Delete(queue, &front, &rear);
            printf("\nDeleted data is: %d\n", data);
            break;
        case 3:
            display(queue, &front, &rear);
            break;
        case 4:
            printf("Exiting...\n");
            break;
        default:
            printf("\nInvalid Choice. Try Again.\n");
            break;
        }

    } while (choice != 4);

    return 0;
}