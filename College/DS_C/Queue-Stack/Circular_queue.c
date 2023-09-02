/**
 * @file Circular_queue.c
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief Circular Queue implementation
 * @version 1.0
 * @date 2022-12-06
 *
 * @copyright Copyright (c) 2022
 *
 */

/*
    Write a program for a circular queue to perform insertion and deletion of data with the help of Add and Delete operation.
*/

#include <stdio.h>
#include <stdlib.h>
#define size 5

int items[size];
int front = -1, rear = -1;

// Check if the queue is full
int isFull()
{
    if ((front == rear + 1) || (front == 0 && rear == size - 1))
        return 1;
    return 0;
}

// Check if the queue is empty
int isEmpty()
{
    if (front == -1)
        return 1;
    return 0;
}

void display()
{
    int i;
    if (isEmpty())
        printf(" \n Empty Queue\n");
    else
    {
        printf("\n Front -> %d ", front);
        printf("\n Items -> ");
        for (i = front; i != rear; i = (i + 1) % size)
        {
            printf("%d ", items[i]);
        }
        printf("%d ", items[i]);
        printf("\n Rear -> %d \n", rear);
    }
}

void enqueue(int element)
{
    if (isFull())
        printf("\n Queue is full!! \n");
    else
    {
        if (front == -1)
            front = 0;
        rear = (rear + 1) % size;
        items[rear] = element;
        printf("\n Inserted -> %d", element);
    }
}

int dequeue()
{
    int element;
    if (isEmpty())
    {
        printf("\n Queue is empty !! \n");
        return (-1);
    }
    else
    {
        element = items[front];
        if (front == rear)
        {
            front = -1;
            rear = -1;
        }
        // Q has only one element, so we reset the
        // queue after dequeing it. ?
        else
        {
            front = (front + 1) % size;
        }
        printf("\n Deleted element -> %d \n", element);
        return (element);
    }
}

int main()
{
    int choice = 0;
    int data = 0;
    do
    {
        printf("\n");
        printf("\n1. Enqueue\n2. Dequeue\n3. Display\n4. Exit\n");
        printf("Enter your choice: ");
        scanf("%d", &choice);
        switch (choice)
        {
        case 1:
            printf("Enter the data to be added: ");
            scanf("%d", &data);
            enqueue(data);
            break;
        case 2:
            data = dequeue();
            printf("\nThe deleted data is: %d", data);
            break;
        case 3:
            display();
            break;
        case 4:
            printf("Exiting...");
            break;
        default:
            printf("Invalid Choice. Try Again.");
            break;
        }
        printf("\n");
    } while (choice != 4);
    return 0;
}