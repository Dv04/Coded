#include <stdio.h>
#include <stdlib.h>

int factorial_iterative(int n)
{
    int i, fact = 1;
    for (i = 1; i <= n; i++)
    {
        fact = fact * i;
    }
    return fact;
}

int factorial_recursive(int n)
{
    if (n == 0)
    {
        return 1;
    }
    else
    {
        return n * factorial_recursive(n - 1);
    }
}

int main()
{
    int n, choice;

    do
    {
        printf("\nEnter the number: ");
        scanf("%d", &n);
        
        printf("Enter 1 for Factorial using iterative method: \n");
        printf("Enter 2 for Factorial using recursive method: \n");
        printf("Enter 3 for Exit: \n");

        printf("\nEnter your choice: ");
        scanf("%d", &choice);


        printf("\n");
        switch (choice)
        {
        case 1:
            printf("Factorial of %d using iterative method is %d", n, factorial_iterative(n));
            break;

        case 2:
            printf("Factorial of %d using recursive method is %d", n, factorial_recursive(n));
            break;

        case 3:
            printf("Exiting...");
            break;

        default:
            printf("Invalid choice");
            break;
        }

    } while (choice != 3);

    return 0;
}