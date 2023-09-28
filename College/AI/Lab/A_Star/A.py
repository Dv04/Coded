# Refactoring the code based on the suggestions

import math
import heapq
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def a_star_tree(start, goal, tree, heuristic):
    """
    A* Algorithm for a tree.
    Returns the path and total cost.
    """
    open_list = []
    closed_list = set()
    start_node = (0, start)
    heapq.heappush(open_list, start_node)
    cost_to_come = {start_node[1]: 0}
    parent_node = {}

    while open_list:
        current_node = heapq.heappop(open_list)
        current_pos = current_node[1]

        if current_pos == goal:
            path = []
            current = current_pos
            while current in parent_node:
                path.append(current)
                current = parent_node[current]
            return path[::-1], cost_to_come[current_pos]

        closed_list.add(current_pos)

        for neighbor, cost in tree[current_pos].items():
            if neighbor in closed_list:
                continue

            new_cost = cost_to_come[current_pos] + cost

            if neighbor not in cost_to_come or new_cost < cost_to_come[neighbor]:
                cost_to_come[neighbor] = new_cost
                total_cost = new_cost + heuristic(neighbor, goal)
                heapq.heappush(open_list, (total_cost, neighbor))
                parent_node[neighbor] = current_pos

    return None, None


def heuristic_tree(node_a, node_b):
    """
    Example heuristic function for the tree. 
    It's a dummy heuristic returning 1 for simplicity.
    """
    return 1

def visualize_tree(tree, path):
    """
    Text-based visualization of the tree and path.
    """
    for node in tree:
        connections = ", ".join([f"{neighbor}({cost})" for neighbor, cost in tree[node].items()])
        if node in path:
            print(f"[{node}] -> {connections}")
        else:
            print(f"{node} -> {connections}")


def main_tree():
    tree = {
        'A': {'B': 2, 'C': 5},
        'B': {'D': 3, 'E': 2},
        'C': {'F': 4, 'G': 2},
        'D': {},
        'E': {'H': 3},
        'F': {},
        'G': {'I': 2},
        'H': {},
        'I': {}
    }
    start_node = 'A'
    goal_node = 'I'

    path, total_cost = a_star_tree(start_node, goal_node, tree, heuristic_tree)

    if path and total_cost is not None:
        visualize_tree(tree, path)
        print(f"\nPath: {path}")
        print(f"Path Length: {len(path)}")
        print(f"Total Cost: {total_cost}\n")
    else:
        print("No path found.")

        
        
# Testing the modified code
# main()
main_tree()

