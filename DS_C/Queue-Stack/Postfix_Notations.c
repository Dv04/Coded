/**
 * @file Polish_Notations.c
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief Polish Notations.
 * @version 1.0
 * @date 2022-10-17
 *
 * @copyright Copyright (c) 2022
 *
 */

/*
    1. Polish Notation:
          - Infix to Postfix
          - Postfix Evaluation

*/

/*
    Logic:
        1. Polish Notation:

            - Infix to Postfix:
                - Scan the infix expression from left to right.
                - If the scanned character is an operand, output it.
                - Else

                    - If the precedence of the scanned operator is greater than the precedence of the operator in the stack (or the stack is empty or the stack contains a ‘(‘ ), push it.

                    - Else, Pop all the operators from the stack which are greater than or equal to in precedence than that of the scanned operator. After doing that Push the scanned operator to the stack. (If you encounter parenthesis while popping then stop there and push the scanned operator in the stack.)

            - Postfix Evaluation:
                - Scan the postfix expression from left to right.
                - If the scanned character is an operand, push it to the stack.
                - Else

                    - Pop the top two values from the stack.
                    - Perform the operation.
                    - Push the result back to the stack.
                - Repeat steps 2-4 until end of expression.


*/

/*
    Write a program to convert infix notations to postfix notations using stack;
*/

#include <stdio.h>

int isOperand(char ch)
{
    if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9'))
        return 1;

    else
        return 0;
}

int isempty(int top)
{
    if (top == -1)
        return 1;

    else
        return 0;
}

int isFull(int top, int size)
{
    if (top == size - 1)
        return 1;

    else
        return 0;
}

void push(char stack[], int *top, char ch)
{
    if (isFull(*top, 100))
        printf("Stack Overflow");

    else if (isempty(*top))
    {
        *top = 0;
        stack[*top] = ch;
    }

    else
    {
        *top = *top + 1;
        stack[*top] = ch;
    }
}

char pop(char stack[], int *top)
{
    char ch;

    if (isempty(*top))
        printf("Stack Underflow");

    else
    {
        ch = stack[*top];
        *top = *top - 1;
    }

    return ch;
}

int precedence(char ch)
{
    if (ch == '+' || ch == '-')
        return 1;

    else if (ch == '*' || ch == '/')
        return 2;

    else if (ch == '^')
        return 3;

    else
        return 0;
}

void InfixToPostfix(char infix[], char postfix[])
{
    char stack[100];
    int top = -1, i = 0, j = 0;
    char ch;

    while (infix[i] != '\0')
    {
        if (isOperand(infix[i]))
        {
            postfix[j] = infix[i];
            j++;
            i++;
        }
        else if (infix[i] == '(')
        {
            push(stack, &top, infix[i]);
            i++;
        }
        else if (infix[i] == ')')
        {
            while (stack[top] != '(')
            {
                postfix[j] = pop(stack, &top);
                j++;
            }
            pop(stack, &top);
            i++;
        }

        else
        {
            if (precedence(infix[i]) > precedence(stack[top]))
                push(stack, &top, infix[i]);

            else if (infix[i] == '^')
            {
                push(stack, &top, infix[i]);
                i++;
            }

            else
            {
                while (precedence(infix[i]) <= precedence(stack[top]))
                {

                    ch = pop(stack, &top);
                    postfix[j] = ch;
                    j++;
                }

                push(stack, &top, infix[i]);
            }

            i++;
        }
    }

    while (!isempty(top))
    {
        ch = pop(stack, &top);
        postfix[j] = ch;
        j++;
    }

    postfix[j] = '\0';
}

int PostfixEvaluation(char postfix[])
{
    char stack[100];
    int top = -1, i = 0;
    int op1, op2, result;

    while (postfix[i] != '\0')
    {
        if (isOperand(postfix[i]))
        {
            push(stack, &top, postfix[i] - '0');
            i++;
        }
        else
        {
            op2 = pop(stack, &top);
            op1 = pop(stack, &top);

            switch (postfix[i])
            {
            case '+':
                result = op1 + op2;
                break;

            case '-':
                result = op1 - op2;
                break;

            case '*':
                result = op1 * op2;
                break;

            case '/':
                result = op1 / op2;
                break;

            case '^':
                result = op1 ^ op2;
                break;
            }

            push(stack, &top, result);
            i++;
        }
    }

    return pop(stack, &top);
}

int main()
{
    char infix[100], postfix[100], prefix[100];

    printf("Enter 1 for converting from infix to postfix\n");
    // printf("Enter 3 for evaluating postfix expression\n");
    printf("Please Enter your Choice: ");

    int choice;
    scanf("%d", &choice);

    switch (choice)
    {
    case 1:
        printf("\n\nEnter the infix expression: ");
        scanf("%s", infix);
        InfixToPostfix(infix, postfix);
        printf("\n\nThe postfix expression is: %s\n\n", postfix);
        break;

    case 3:
        printf("\n\nEnter the postfix expression: ");
        scanf("%s", postfix);
        printf("\n\nThe result of the postfix expression is: %d\n\n", PostfixEvaluation(postfix));
        break;

    default:
        printf("\n\nInvalid Choice\n\n");
        break;
    }
    return 0;
}

// ((a+b^C^d)*(e+f/d))