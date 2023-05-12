/**
 * @file Stack.c
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief Stack Homework.
 * @version 1.0
 * @date 2022-10-10
 *
 * @copyright Copyright (c) 2022
 *
 */

/*
    Logic:
*/

/**
 * Write a program for an stack to push "tiger&lion" and pop "tiger&lion" with the help of push and pop operation.
 * Also display the stack after writing the program.
 */

#include <stdio.h>

int isFull(int *top, int size)
{
    if (*top == size - 1)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

int isEmpty(int *top)
{
    if (*top == -1)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

void display(char *stack, int *top)
{
    if (isEmpty(top))
    {
        printf("\nStack is Empty. Cannot Display.\n");
    }
    else
    {
        for (int i = *top; i >= 0; i--)
        {
            printf("%c\n", stack[i]);
        }
    }
}

void push(char stack[], int *top, int size, char data)
{
    if (isFull(top, size))
    {
        printf("\nStack is Full. Cannot add.\n");
    }
    else
    {
        printf("\nStack is not Full.\n");
        *top = *top + 1;
        stack[*top] = data;
    }

    printf("\n");
    display(stack, top);
}

void pop(char stack[], int *top, int size)
{
    char data;
    if (isEmpty(top))
    {
        printf("\nStack is Empty. Cannot pop.\n");
    }
    else
    {
        printf("\nStack is not Empty.\n");
        data = stack[*top];
        *top = *top - 1;
        printf("\nPopped Data: %c\n", data);
    }

    printf("\n");
    if (!isEmpty(top))
    {
        display(stack, top);
    }
}

int main()
{
    char stack[1000];
    printf("The program is starting\n");
    int top = -1;
    int *ptr = &top;
    int choice;
    int tp;

    do
    {
        printf("\n");
        printf("Enter 1 for push, 2 for pop, 3 for exit: ");
        scanf("%d", &choice);
        printf("\n");
        switch (choice)
        {
        case 1:
            printf("Enter the data to be pushed: ");
            char data;
            scanf(" %c", &data);
            push(stack, ptr, 1000, data);
            break;

        case 2:
            printf("The last data will be popped\n");
            pop(stack, ptr, 1000);
            break;

        case 3:
            printf("The program is exiting\n\n");
            break;

        default:

            printf("Wrong input.");
            break;
        }
    } while (choice != 3);

    return 0;
}