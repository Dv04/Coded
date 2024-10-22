import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import networkx as nx
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# Load the dataset
data = pd.read_csv("loan_approval_dataset.csv")

# Drop the loan_status column
data = data.drop(columns=["loan_status"], axis=1)

# Label encode 'education' and 'self_employed' columns
data["education"] = data["education"].map({" Not Graduate": 0, " Graduate": 1})
data["self_employed"] = data["self_employed"].map({" No": 0, " Yes": 1})

# Separate features and target variable
X = data.drop(columns=["loan_amount"])
y = data["loan_amount"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Scale the target variable as well
y_scaler = StandardScaler()
y_train = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test = y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()

# Initialize a graph
G = nx.Graph()

# Add nodes with features
for index, row in tqdm(data.iterrows(), total=len(data), desc="Adding Nodes"):
    G.add_node(
        index,
        no_of_dependents=row["no_of_dependents"],
        education=row["education"],
        self_employed=row["self_employed"],
        income_annum=row["income_annum"],
        loan_amount=row["loan_amount"],
        loan_term=row["loan_term"],
        cibil_score=row["cibil_score"],
        residential_assets_value=row["residential_assets_value"],
        commercial_assets_value=row["commercial_assets_value"],
        luxury_assets_value=row["luxury_assets_value"],
        bank_asset_value=row["bank_asset_value"],
    )

# Add edges based on criteria
for index, row in tqdm(data.iterrows(), total=len(data), desc="Adding Edges"):
    for other_index, other_row in data.iterrows():
        if index != other_index:
            if abs(row["income_annum"] - other_row["income_annum"]) < 1000000:
                G.add_edge(index, other_index)


# Convert graph to PyTorch Geometric data
def convert_graph_to_pyg_data(G):
    # Create a list to hold node features
    node_features = []
    for node in G.nodes:
        features = [
            G.nodes[node].get("no_of_dependents", 0),
            G.nodes[node].get("education", 0),
            G.nodes[node].get("self_employed", 0),
            G.nodes[node].get("income_annum", 0),
            G.nodes[node].get("loan_term", 0),
            G.nodes[node].get("cibil_score", 0),
            G.nodes[node].get("residential_assets_value", 0),
            G.nodes[node].get("commercial_assets_value", 0),
            G.nodes[node].get("luxury_assets_value", 0),
            G.nodes[node].get("bank_asset_value", 0),
        ]
        node_features.append(features)
    node_features = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
    return Data(x=node_features, edge_index=edge_index)


# Create PyG data object
data_pyg = convert_graph_to_pyg_data(G)


# Define the GNN model
class GNNModel(torch.nn.Module):
    def __init__(self, input_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, 64)  # First GCN layer
        self.conv2 = GCNConv(64, 32)  # Second GCN layer
        self.fc = torch.nn.Linear(32, 1)  # Fully connected layer for output

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))  # First layer with ReLU
        x = F.relu(self.conv2(x, edge_index))  # Second layer with ReLU
        x = self.fc(x)  # Fully connected layer
        return x


# Instantiate the model
input_dim = data_pyg.x.shape[1]  # Number of features
model = GNNModel(input_dim)

# Define loss function and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()

# Train the model
model.train()
for epoch in range(100):  # Number of epochs
    optimizer.zero_grad()
    out = model(data_pyg)
    out_train = out[: len(y_train)]  # Make sure out matches y_train size
    y_train_tensor = torch.tensor(y_train, dtype=torch.float)  # Convert to tensor
    loss = loss_fn(
        out_train.flatten(), y_train_tensor
    )  # Calculate loss based on matching sizes
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Evaluate the model
model.eval()
y_pred = model(data_pyg).detach().numpy()
y_pred = y_scaler.inverse_transform(y_pred).flatten()  # Inverse scale predictions

# Custom input data
custom_input = pd.DataFrame(
    {
        "loan_id": [101, 102, 103, 104],
        "no_of_dependents": [2, 5, 3, 0],
        "education": [" Graduate", " Not Graduate", " Graduate", " Graduate"],
        "self_employed": [" No", " Yes", " No", " No"],
        "income_annum": [3900000, 1200000, 5000000, 300000],
        "loan_amount": [12300000, 5000000, 1500000, 10000000],
        "loan_term": [18, 12, 24, 18],
        "cibil_score": [700, 600, 750, 800],
        "residential_assets_value": [7600000, 200000, 10000000, 5000000],
        "commercial_assets_value": [690000, 1000000, 500000, 3000000],
        "luxury_assets_value": [1300000, 200000, 10000, 5000000],
        "bank_asset_value": [2800000, 50000, 200000, 300000],
    }
)

# Label encode 'education' and 'self_employed' columns
custom_input["education"] = custom_input["education"].map(
    {" Not Graduate": 0, " Graduate": 1}
)
custom_input["self_employed"] = custom_input["self_employed"].map({" No": 0, " Yes": 1})

# Standardize the features
X_custom = custom_input.drop(columns=["loan_amount"])
X_custom = scaler.transform(X_custom)

# Update graph with custom input
custom_graph = nx.Graph()
for index, row in tqdm(
    custom_input.iterrows(), total=len(custom_input), desc="Adding Custom Nodes"
):
    custom_graph.add_node(
        index,
        no_of_dependents=row["no_of_dependents"],
        education=row["education"],
        self_employed=row["self_employed"],
        income_annum=row["income_annum"],
        loan_amount=row["loan_amount"],
        loan_term=row["loan_term"],
        cibil_score=row["cibil_score"],
        residential_assets_value=row["residential_assets_value"],
        commercial_assets_value=row["commercial_assets_value"],
        luxury_assets_value=row["luxury_assets_value"],
        bank_asset_value=row["bank_asset_value"],
    )

# Adding custom edges for the new graph
for index, row in tqdm(
    custom_input.iterrows(), total=len(custom_input), desc="Adding Custom Edges"
):
    for other_index, other_row in custom_input.iterrows():
        if index != other_index:
            if (
                abs(row["income_annum"] - other_row["income_annum"]) < 1000000
            ):  # Example threshold
                custom_graph.add_edge(index, other_index)

# Convert custom graph to PyTorch Geometric data
custom_data_pyg = convert_graph_to_pyg_data(custom_graph)

# Predict the loan amount for custom input
y_custom_pred = model(custom_data_pyg).detach().numpy()
y_custom_pred = y_scaler.inverse_transform(y_custom_pred).flatten()

# Display predictions and comparison with actual values
print(f"\n{'='*70}\n{'Loan Approval Predictions':^70}\n{'='*70}")
for i in range(len(y_custom_pred)):
    approval_status = "Yes" if y_custom_pred[i] > X_custom.iloc[i] else "No"
    print(
        f"Loan Approval: {approval_status} (Predicted Amount: {y_custom_pred[i]:.2f}, Actual Amount: {X_custom.iloc[i]:.2f})"
    )
