def tsp(graph):
    n = len(graph)
    # memoization table, where dp[mask][i] stores the minimum cost to visit cities in mask, ending in city i
    dp = [[float('inf')] * n for _ in range(1 << n)]
    dp[1][0] = 0  # starting from city 0

    # Iterate through all subsets of vertices
    for mask in range(1, 1 << n):
        for u in range(n):
            # Continue if u is not in the subset represented by mask
            if not (mask & (1 << u)):
                continue

            # Try to find the shortest path to u from any vertex v
            for v in range(n):
                if mask & (1 << v) and u != v:
                    dp[mask][u] = min(dp[mask][u], dp[mask ^ (1 << u)][v] + graph[v][u])

    # Reconstruct the shortest path and compute the minimum cost
    mask = (1 << n) - 1  # All cities have been visited
    u = 0
    min_cost = float('inf')
    for v in range(1, n):
        if dp[mask][v] + graph[v][0] < min_cost:
            min_cost = dp[mask][v] + graph[v][0]
            u = v

    # Reconstruct the path
    path = []
    for _ in range(n - 1):
        path.append(u)
        mask ^= (1 << u)
        v = u
        for u in range(n):
            if mask & (1 << u) and dp[mask][u] + graph[u][v] == dp[mask ^ (1 << v)][v] + graph[v][u]:
                break
    path.append(0)
    path.reverse()

    return min_cost, path

# Example usage
graph = [
    [0, 29, 20, 21],
    [29, 0, 15, 17],
    [20, 15, 0, 28],
    [21, 17, 28, 0]
]

min_cost, path = tsp(graph)
print(f"Minimum cost and path of TSP: {min_cost}, {path}")