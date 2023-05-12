/**
 * @file Prim.c
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief Prim's Algorithm using iterative and recursive algorithms
 * @version 1.0
 * @date 2023-01-17
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#define V 5
#define MAX_VERTICES 100
#define INF 2147483647

int minKey(int key[], bool mstSet[])
{
    int min = INF, min_index;

    for (int v = 0; v < V; v++)
    {
        if (mstSet[v] == false && key[v] < min)
        {
            min = key[v], min_index = v;
        }
    }
    return min_index;
}

void primMST(int graph[V][V])
{
    int parent[V];
    int key[V];
    bool mstSet[V];
    for (int i = 0; i < V; i++)
    {
        key[i] = INT_MAX;
        mstSet[i] = false;
    }
    key[0] = 0;
    parent[0] = -1;
    for (int count = 0; count < V - 1; count++)
    {
        int u = minKey(key, mstSet);

        mstSet[u] = true;
        for (int v = 0; v < V; v++)
        {
            if (graph[u][v] && mstSet[v] == false && graph[u][v] < key[v])
            {
                parent[v] = u, key[v] = graph[u][v];
            }
        }
    }
    printf("Edge\tWeight\n");
    for (int i = 1; i < V; i++)
    {
        printf("%d - %d \t%d \n", parent[i], i, graph[i][parent[i]]);
    }
}

int main()
{
    int graph[V][V] = {{0, 0, 3, 0, 0},
                       {0, 0, 10, 4, 0},
                       {3, 10, 0, 2, 6},
                       {0, 4, 2, 0, 1},
                       {0, 0, 6, 1, 0}};

    primMST(graph);

    return 0;
}
