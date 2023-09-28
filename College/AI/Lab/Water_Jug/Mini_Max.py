from collections import deque

def actions(state, m, n):
    """
    Generate all possible actions from the current state.
    """
    a, b = state
    return [
        ('fill', (m, b), 'Fill A'),
        ('fill', (a, n), 'Fill B'),
        ('empty', (0, b), 'Empty A'),
        ('empty', (a, 0), 'Empty B'),
        ('pour', (max(a + b - n, 0), min(n, a + b)), 'Pour A to B'),
        ('pour', (min(m, a + b), max(a + b - m, 0)), 'Pour B to A')
    ]


def water_jug_minimax(m, n, target):
    """
    Solve the water jug problem using the Mini-Max algorithm.
    """
    start = (0, 0)
    queue = deque([(start, [])])
    visited = set()
    solutions = []

    while queue:
        state, path = queue.popleft()
        if state in visited:
            continue
        visited.add(state)

        # Check for goal state
        a, b = state
        if a == target or b == target:
            solutions.append(path + [state])

        for action, next_state, action_description in actions(state, m, n):
            if next_state not in visited:
                new_path = path + [state, action_description]
                queue.append((next_state, new_path))

    return solutions if solutions else None


# Testing the modified code with Mini-Max algorithm
m, n, target = 4, 3, 2
solutions = water_jug_minimax(m, n, target)
for i, solution in enumerate(solutions, 1):
    print(f"Solution {i}:")
    for step in solution:
        print(step)
