/**
 * @file Kruskal.c
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief Krushkal's Algorithm
 * @version 1.0
 * @date 2023-01-17
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <stdio.h>
#include <stdlib.h>

#define MAX_VERTICES 100
#define INF 2147483647

typedef struct edge
{
    int u, v, weight;
} Edge;

typedef struct Graph
{
    int n, m;
    Edge edges[MAX_VERTICES];
} Graph;

void init_graph(Graph *G, int n)
{
    G->n = n;
    G->m = 0;
}

void add_edge(Graph *G, int u, int v, int weight)
{
    G->edges[G->m].u = u;
    G->edges[G->m].v = v;
    G->edges[G->m].weight = weight;
    G->m++;
}

int parent[MAX_VERTICES];

int find_root(int u)
{
    if (u != parent[u])
        parent[u] = find_root(parent[u]);
    return parent[u];
}

void Kruskal(Graph *G)
{
    int e = 0;
    int i, j;
    printf("\n\n");
    printf("V1\tV2\tweight: \n");
    for (i = 1; i <= G->n; i++)
    {
        parent[i] = i;
    }
    while (e < G->n - 1)
    {
        int min = INF;
        int u, v;
        for (i = 0; i < G->m; i++)
        {
            if (find_root(G->edges[i].u) != find_root(G->edges[i].v) && G->edges[i].weight < min)
            {
                min = G->edges[i].weight;
                u = G->edges[i].u;
                v = G->edges[i].v;
            }
        }
        if (min != INF)
        {
            printf("%d\t%d\t%d", u, v, min);
            printf("\n");
            parent[find_root(u)] = find_root(v);
            e++;
        }
    }
}

int main()
{
    int n, m, u, v, w;
    Graph G;
    printf("Enter the number of Vertices: ");
    scanf("%d", &n);
    init_graph(&G, n);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("weight between %d and %d: ", i, j);
            scanf("%d", &w);
            add_edge(&G, i, j, w);
        }
    }
    Kruskal(&G);
    return 0;
}