/**
 * @file Prefix_Notations.c
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief Reverse Polish Notations
 * @version 1.0
 * @date 2022-10-17
 *
 * @copyright Copyright (c) 2022
 *
 */

/*
        2. Reverse Polish Notation:

            - Infix to Prefix:
                - Scan the infix expression from right to left.
                - If the scanned character is an operand, output it.
                - Else,
                    - If the precedence of the scanned operator is greater than the precedence of the operator in the stack(or the stack is empty or the stack contains a ‘)’ ), push it.
                    - Else, Pop all the operators from the stack which are greater than or equal to in precedence than that of the scanned operator. After doing that Push the scanned operator to the stack. (If you encounter parenthesis while popping then stop there and push the scanned operator in the stack.)

            - Prefix Evaluation:
                - Scan the prefix expression from right to left.
                - If the scanned character is an operand, push it to the stack.
                - Else,
                    - Pop the top two values from the stack.
                    - Perform the operation.
                    - Push the result back to the stack.
                - Repeat steps 2-4 until end of expression.
*/

#include <stdio.h>
#include <string.h>

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

void push(char stack[], int *top, char ch)
{
    if (isFull(*top, 100))
    {
        printf("Stack Overflow!");
    }
    else if (ch == '(')
        stack[++(*top)] = ')';
    else if (ch == ')')
        stack[++(*top)] = '(';
    else
        stack[++(*top)] = ch;
}

char pop(char stack[], int *top)
{
    if (isempty(*top))
    {
        printf("Stack Underflow!");
        return -1;
    }
    else
        return stack[(*top)--];
}

void strrev(char str[])
{
    int i, j;
    char temp;
    for (i = 0, j = strlen(str) - 1; i < j; i++, j--)
    {
        temp = str[i];
        str[i] = str[j];
        str[j] = temp;
    }
}

void InfixToPrefix(char infix[], char prefix[])
{
    char stack[100];
    int top = -1, i, j = 0;
    for (i = strlen(infix) - 1; i >= 0; i--)
    {
        if (isOperand(infix[i]))
        {
            prefix[j++] = infix[i];
        }
        else if (infix[i] == ')')
        {
            push(stack, &top, infix[i]);
        }
        else if (infix[i] == '(')
        {
            while (stack[top] != ')')
            {
                prefix[j++] = pop(stack, &top);
            }
            pop(stack, &top);
        }
        else
        {
            if (precedence(infix[i]) > precedence(stack[top]))
            {
                push(stack, &top, infix[i]);
            }
            else
            {
                while (precedence(infix[i]) <= precedence(stack[top]))
                {
                    prefix[j++] = pop(stack, &top);
                }
                push(stack, &top, infix[i]);
            }
        }
    }
    while (!isempty(top))
    {
        prefix[j++] = pop(stack, &top);
    }
    prefix[j] = '\0';
    strrev(prefix);
}

int PrefixEvaluation(char prefix[]);

int main()
{
    char infix[100], postfix[100], prefix[100];

    printf("Enter 1 for converting from infix to prefix\n");
    // printf("Enter 2 for evaluating postfix expression\n");
    printf("Please Enter your Choice: ");

    int choice;
    scanf("%d", &choice);

    switch (choice)
    {

    case 1:
        printf("\n\nEnter the infix expression: ");
        scanf("%s", infix);
        InfixToPrefix(infix, prefix);
        printf("\n\nThe prefix expression is: %s\n\n", prefix);
        break;

    case 2:
        printf("\n\nEnter the prefix expression: ");
        scanf("%s", prefix);
        printf("\n\nThe result of the prefix expression is: %d\n\n", PrefixEvaluation(prefix));
        break;

    default:
        printf("\n\nInvalid Choice\n\n");
        break;
    }
    return 0;
}

int PrefixEvaluation(char prefix[])
{
    char stack[100];
    int top = -1, i, j, k, num, result;
    for (i = strlen(prefix) - 1; i >= 0; i--)
    {
        if (isOperand(prefix[i]))
        {
            num = 0;
            j = 0;
            k = 1;
            while (isOperand(prefix[i]))
            {
                num += (prefix[i] - '0') * k;
                k *= 10;
                i--;
                j++;
            }
            i++;
            push(stack, &top, num);
        }
        else
        {
            int op1 = pop(stack, &top);
            int op2 = pop(stack, &top);
            switch (prefix[i])
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
        }
    }
    return pop(stack, &top);
}