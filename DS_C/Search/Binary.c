/**
 * @file Binary.c
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief Binary Search Tree
 * @version 1.0
 * @date 2023-01-17
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <stdio.h>
#include <stdlib.h>

typedef struct node
{
    int data;
    struct node *left;
    struct node *right;
} node;

node *createNode(int data)
{
    node *newNode = (node *)malloc(sizeof(node));
    newNode->data = data;
    newNode->left = NULL;
    newNode->right = NULL;
    return newNode;
}

node *insert(node *root, int data)
{
    if (root == NULL)
    {
        root = createNode(data);
    }
    else if (data <= root->data)
    {
        root->left = insert(root->left, data);
    }
    else
    {
        root->right = insert(root->right, data);
    }
    return root;
}

int search(node *root, int data)
{
    if (root == NULL)
    {
        return 0;
    }
    else if (root->data == data)
    {
        return 1;
    }
    else if (data <= root->data)
    {
        return search(root->left, data);
    }
    else
    {
        return search(root->right, data);
    }
}

int printli(node *root)
{
    if (root == NULL)
    {
        return 0;
    }
    printli(root->left);
    printf("%d ", root->data);
    printli(root->right);
    return 0;
}

int main()
{
    node *root = NULL;
    root = insert(root, 15);
    root = insert(root, 10);
    root = insert(root, 20);
    root = insert(root, 25);
    root = insert(root, 8);
    root = insert(root, 12);
    printli(root);
    printf("\n");
    int number;
    printf("Enter number be searched : ");
    scanf("%d", &number);
    if (search(root, number) == 1)
    {
        printf("Found %d.\n", number);
    }
    else
    {
        printf("Not found %d.\n", number);
    }
    return 0;
}
