/**
 * @file tree.c
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief Tree Traversal
 * @version 1.0
 * @date 2022-12-19
 *
 * @copyright Copyright (c) 2022
 *
 */

// Implementation of binary tree and its traversal (preorder, inorder, postorder).

/*
    Logic:
        1. Insert the data in the tree.
        2. Traverse the tree in preorder, inorder and postorder.
            2.1 Use Recursion to Traverse the tree.
        3. Print the tree.

*/

#include <stdio.h>
#include <stdlib.h>
#define COUNT 10

struct node
{
    int data;
    struct node *left;
    struct node *right;
};
struct node *root = NULL;

void insert(int data)
{
    struct node *new_node;
    new_node = (struct node *)malloc(sizeof(struct node));
    new_node->data = data;
    new_node->left = NULL;
    new_node->right = NULL;

    if (root == NULL)
    {
        root = new_node;
    }
    else
    {
        struct node *temp;
        temp = root;
        while (1)
        {
            if (data < temp->data)
            {
                if (temp->left == NULL)
                {
                    temp->left = new_node;
                    break;
                }
                else
                {
                    temp = temp->left;
                }
            }
            else
            {
                if (temp->right == NULL)
                {
                    temp->right = new_node;
                    break;
                }
                else
                {
                    temp = temp->right;
                }
            }
        }
    }
}

void print(struct node *root, int space)
{
    if (root == NULL)
        return;

    space += COUNT;

    print(root->right, space);

    printf("\n");
    for (int i = COUNT; i < space; i++)
        printf(" ");
    printf("%d\n", root->data);

    print(root->left, space);
}

void preorder(struct node *temp)
{
    if (temp != NULL)
    {
        printf("%d ", temp->data);
        preorder(temp->left);
        preorder(temp->right);
    }
}

void inorder(struct node *temp)
{
    if (temp != NULL)
    {
        inorder(temp->left);
        printf("%d ", temp->data);
        inorder(temp->right);
    }
}

void postorder(struct node *temp)
{
    if (temp != NULL)
    {
        postorder(temp->left);
        postorder(temp->right);
        printf("%d ", temp->data);
    }
}

int main()
{
    int choice, data;
    printf("Press\n1 to Insert\n2 to Preorder\n3 to Inorder\n4 to Postorder\n5 to Print as a tree\n6 to Exit\n");
    while (1)
    {
        printf("Enter your choice: ");
        scanf("%d", &choice);
        switch (choice)
        {
        case 1:
            printf("Enter the data: ");
            scanf("%d", &data);
            insert(data);
            print(root, 0);
            break;
        case 2:
            if (root == NULL)
            {
                printf("Tree is empty.\n");
                break;
            }
            else
            {
                printf("Preorder: ");
                preorder(root);
            }
            printf("\n");
            break;
        case 3:
            if (root == NULL)
            {
                printf("Tree is empty.\n");
                break;
            }
            else
            {
                printf("\n\nInorder: ");
                inorder(root);
            }
            printf("\n\n");
            break;
        case 4:
            if (root == NULL)
            {
                printf("Tree is empty.\n");
                break;
            }
            else
            {
                printf("\n\nPostorder: ");
                postorder(root);
            }
            printf("\n\n");
            break;
        case 5:
            print(root, 0);
            break;
        case 6:
            exit(0);
            break;
        default:
            printf("Invalid choice.\n");
            break;
        }
    }
    return 0;
}