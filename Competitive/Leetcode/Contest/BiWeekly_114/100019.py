from collections import defaultdict

def max_num_components(n, edges, values, k):
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)  # it's an undirected graph
        
    total_values = {}
    visited = set()

    def dfs(node):
        visited.add(node)
        total_values[node] = values[node]
        for neighbor in graph[node]:
            if neighbor not in visited:
                total_values[node] += dfs(neighbor)
                graph[neighbor].remove(node)  # remove back edge to make it a tree
        return total_values[node]

    total_value = dfs(0)
    count = 0

    for node in total_values:
        if (total_values[node] % k == 0) and ((total_value - total_values[node]) % k == 0):
            count += 1

    return count

# Testing the function with the given examples
n1, edges1, values1, k1 = 5, [[0,2],[1,2],[1,3],[2,4]], [1,8,1,4,4], 6
output1 = max_num_components(n1, edges1, values1, k1)

n2, edges2, values2, k2 = 7, [[0,1],[0,2],[1,3],[1,4],[2,5],[2,6]], [3,0,6,1,5,2,1], 3
output2 = max_num_components(n2, edges2, values2, k2)

print(output1, output2)
