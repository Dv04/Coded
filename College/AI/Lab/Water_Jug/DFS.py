from typing import Set, Tuple, List

class State:
    def __init__(self, a: int, b: int):
        self.a = a
        self.b = b

def dfs(a: int, b: int, target: int) -> bool:
    visited: Set[Tuple[int, int]] = set()
    stack: List[State] = [State(0, 0)]

    while stack:
        current_state: State = stack.pop()
        if (current_state.a, current_state.b) in visited:
            continue
        visited.add((current_state.a, current_state.b))

# Check if the target state is reached
        if current_state.a == target or current_state.b == target:
            print(f"Solution found: ({current_state.a}, {current_state.b})")
            return True

        # Empty Jug A
        print(f"Emptying jug A from ({current_state.a}, {current_state.b})")
        stack.append(State(0, current_state.b))

        # Empty Jug B
        print(f"Emptying jug B from ({current_state.a}, {current_state.b})")
        stack.append(State(current_state.a, 0))

        # Fill Jug A
        print(f"Filling jug A from ({current_state.a}, {current_state.b})")
        stack.append(State(a, current_state.b))

        # Fill Jug B
        print(f"Filling jug B from ({current_state.a}, {current_state.b})")
        stack.append(State(current_state.a, b))

        # Pour from Jug A to Jug B
        pour_a_to_b = min(current_state.a, b - current_state.b)
        print(f"Pouring from jug A to B from ({current_state.a}, {current_state.b})")
        stack.append(State(current_state.a - pour_a_to_b, current_state.b + pour_a_to_b))

        # Pour from Jug B to Jug A
        pour_b_to_a = min(current_state.b, a - current_state.a)
        print(f"Pouring from jug B to A from ({current_state.a}, {current_state.b})")
        stack.append(State(current_state.a + pour_b_to_a, current_state.b - pour_b_to_a))
    print("No solution found.")
    return False

if __name__ == '__main__':
    a = int(input("Enter the capacity of jug A: "))
    b = int(input("Enter the capacity of jug B: "))
    target = int(input("Enter the target amount: "))
    dfs(a, b, target)
