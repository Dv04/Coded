/**
 * @file Stack.c
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief Stack Applications.
 * @version 1.0
 * @date 2022-10-10
 *
 * @copyright Copyright (c) 2022
 *
 */

/* Write a program for a stack to perform insertion and deletion of data with the help of push and pop operation.
 */

/*

    Logic:
        1. Start
        2. Declare variables and array of size given by user. This array will be used as stack.
        3. Declare top variable and initialize it to -1 and a pointer ptr to point to the top.
        4. Declare a variable choice to store the choice of user.
        5. Display the menu to user and ask for choice.
        6. If choice is 1, then ask for data to be inserted and call push function.
            6.1 Check is Stack is full or not.
            6.2 If full, display message and return.
            6.3 If not full, increment top by 1 and store data at top.
            6.4 Display the stack.
            6.5 Exit the Function.
        7. If choice is 2, then call pop function.
            7.1 Check if Stack is empty or not.
            7.2 If empty, display message and return.
            7.3 If not empty, store the data at top in a variable and decrement top by 1.
            7.4 Display the stack.
            7.5 Exit the Function.
        8. If choice is 3, then exit.
        9. If choice is not 1, 2 or 3, then display error message.
        10. End

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

void display(int *stack, int *top)
{
    if (isEmpty(top))
    {
        printf("\nStack is Empty. Cannot Display.\n");
    }
    else
    {
        for (int i = *top; i >= 0; i--)
        {
            printf("%d\n", stack[i]);
        }
    }
}

void push(int stack[], int *top, int size, int data)
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

void pop(int stack[], int *top, int size)
{
    int data;
    if (isEmpty(top))
    {
        printf("\nStack is Empty. Cannot pop.\n");
    }
    else
    {
        printf("\nStack is not Empty.\n");
        data = stack[*top];
        *top = *top - 1;
        printf("\nPopped Data: %d\n", data);
    }

    printf("\n");
    display(stack, top);
}

int main()
{
    int stack[5];
    printf("The program is starting\n");
    int top = -1;
    int *ptr = &top;
    int choice;

    do
    {
        printf("\n");
        printf("Enter 1 for push, 2 for pop,3 for display, 4 for exit: ");
        scanf("%d", &choice);
        switch (choice)
        {
        case 1:
            printf("Enter the data to be pushed: ");
            int data;
            scanf("%d", &data);
            push(stack, ptr, 5, data);
            break;

        case 2:
            printf("The last data will be popped\n");
            pop(stack, ptr, 5);
            break;
        case 3:
            printf("Displaying the data.\n");
            display(stack, ptr);
            break;
        case 4:
            printf("The program is exiting\n\n");
            break;

        default:

            printf("Wrong input.");
            break;
        }
    } while (choice != 4);

    return 0;
}