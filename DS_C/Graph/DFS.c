/**
 * @file DFS.c
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief Depth First Search in Graph
 * @version 1.0
 * @date 2023-01-17
 * 
 * @copyright Copyright (c) 2023
 * 
 */



#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define MAX_VERTICES 100
#define INF 2147483647

typedef struct
{
    int n, m;
    int A[MAX_VERTICES][MAX_VERTICES];
} Graph;

void init_graph(Graph *G, int n)
{
    G->n = n;
    G->m = 0;
    int i, j;
    for (i = 1; i <= n; i++)
        for (j = 1; j <= n; j++)
            G->A[i][j] = 0;
}

void add_edge(Graph *G, int x, int y)
{
    G->A[x][y] = 1;
    G->A[y][x] = 1;
}

int mark[MAX_VERTICES];

void DFS(Graph *G, int x)
{
    int stack[MAX_VERTICES];
    int top = -1;
    printf("%d ", x);
    mark[x] = 1;
    stack[++top] = x;
    while (top != -1)
    {
        int u = stack[top];
        int v;
        for (v = 1; v <= G->n; v++)
        {
            if (G->A[u][v] != 0 && mark[v] == 0)
            {
                printf("%d ", v);
                mark[v] = 1;
                stack[++top] = v;
                break;
            }
        }
        if (u == stack[top])
            top--;
    }
}

int main()
{
    Graph G;
    int n, m, u, v, e;
    printf("Enter number of vertices: ");
    scanf("%d", &n);
    init_graph(&G, n);
    printf("Enter number of edges: ");
    scanf("%d", &m);
    for (e = 1; e <= m; e++)
    {
        printf("Enter edge %d (format: u v): ", e);
        scanf("%d%d", &u, &v);
        add_edge(&G, u, v);
    }
    printf(" DFS: ");
    DFS(&G, 1);
    printf("\n");
    return 0;
}
