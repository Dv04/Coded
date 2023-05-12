class Solution:
    def longestCycle(self, edges: list[int]) -> int:
        def dfs(node, visited, depth):
            nonlocal max_cycle_len

            if visited[node] != 0:
                # A cycle is found
                if visited[node] == 1:
                    max_cycle_len = max(max_cycle_len, depth - depths[node])
                return

            visited[node] = 1
            depths[node] = depth

            if edges[node] != -1:
                dfs(edges[node], visited, depth + 1)

            visited[node] = 2

        n = len(edges)
        visited = [0] * n  # 0: unvisited, 1: visiting, 2: visited
        depths = [0] * n
        max_cycle_len = -1

        for i in range(n):
            if visited[i] == 0:
                dfs(i, visited, 1)

        return max_cycle_len

    
print(Solution().longestCycle([3,3,4,2,3]))