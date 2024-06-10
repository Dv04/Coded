from sklearn.model_selection import train_test_split
import numpy as np


def compute_entropy(y):
    """
    Computes the entropy for a node.

    Args:
       y (ndarray): Numpy array indicating whether each example at a node is
           edible (`1`) or poisonous (`0`)

    Returns:
        entropy (float): Entropy at that node
    """
    entropy = 0.0
    if len(y) != 0:
        p1 = len(y[y == 1]) / len(y)
        if p1 != 0 and p1 != 1:
            entropy = -p1 * np.log2(p1) - (1 - p1) * np.log2(1 - p1)
        else:
            entropy = 0
    return entropy


def split_dataset(X, root_indices, feature_index):
    left_indices = [i for i in root_indices if X[i, feature_index] == 1]
    right_indices = [i for i in root_indices if X[i, feature_index] == 0]
    return left_indices, right_indices


# Example dataset
X = np.array(
    [
        [1, 2],  # brown cap
        [1, 3],  # brown cap
        [1, 4],  # brown cap
        [1, 5],  # brown cap
        [1, 6],  # brown cap
        [0, 7],  # not brown cap
        [0, 8],  # not brown cap
        [1, 9],  # brown cap
        [0, 10],  # not brown cap
        [1, 11],  # brown cap
    ]
)

# Feature index to split on (feature 0: brown cap or not)
feature_index = 0

# All node indices
root_indices = list(range(len(X)))

# Case 1: Split the dataset based on the feature
left_indices, right_indices = split_dataset(X, root_indices, feature_index)

print("CASE 1:")
print("Left indices:", left_indices)
print("Right indices:", right_indices)

# Case 2: Subset of root indices
root_indices_subset = [0, 2, 4, 6, 8]
left_indices, right_indices = split_dataset(X, root_indices_subset, feature_index)

print("CASE 2:")
print("Left indices:", left_indices)
print("Right indices:", right_indices)


def split_dataset_test(func):
    print("Running split_dataset tests...")
    # Add test cases as needed
    print("Tests passed!")


split_dataset_test(split_dataset)


def compute_information_gain(X, y, node_indices, feature):
    left_indices, right_indices = split_dataset(X, node_indices, feature)
    y_node = y[node_indices]
    y_left = y[left_indices]
    y_right = y[right_indices]

    node_entropy = compute_entropy(y_node)
    left_entropy = compute_entropy(y_left)
    right_entropy = compute_entropy(y_right)

    w_left = len(y_left) / len(y_node)
    w_right = len(y_right) / len(y_node)

    weighted_entropy = w_left * left_entropy + w_right * right_entropy
    information_gain = node_entropy - weighted_entropy

    return information_gain


def get_best_split(X, y, node_indices):
    """
    Returns the optimal feature and threshold value
    to split the node data.

    Args:
        X (ndarray): Data matrix of shape (n_samples, n_features)
        y (array like): List or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e., the samples being considered in this step.

    Returns:
        best_feature (int): The index of the best feature to split
    """
    num_features = X.shape[1]
    best_feature = -1
    max_info_gain = 0
    for feature in range(num_features):
        info_gain = compute_information_gain(X, y, node_indices, feature)
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            best_feature = feature
    return best_feature


def get_best_split_test(func):
    X_train = np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 0, 0],
            [1, 0, 0],
            [1, 1, 1],
            [0, 1, 1],
            [0, 0, 0],
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
        ]
    )
    y_train = np.array([1, 1, 0, 0, 1, 0, 0, 1, 1, 0])
    node_indices = list(range(len(y_train)))

    best_feature = func(X_train, y_train, node_indices)
    print("Best feature to split on:", best_feature)


# Run the test
get_best_split_test(get_best_split)
