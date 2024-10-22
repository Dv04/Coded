import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
from termcolor import colored
import numpy as np

# Load the dataset
data = pd.read_csv("loan_approval_dataset.csv")

# Drop the loan_status column
data = data.drop(columns=["loan_status"], axis=1)

# Label encode 'education' and 'self_employed' columns
data["education"] = data["education"].map({" Not Graduate": 0, " Graduate": 1})
data["self_employed"] = data["self_employed"].map({" No": 0, " Yes": 1})

# Separate features and target variable
X = data.drop(columns=["loan_id", "loan_amount"])
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
y_train = y_scaler.fit_transform(
    y_train.values.reshape(-1, 1)
).flatten()  # Scaling the target variable
y_test = y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()

# GAN Model Building


# Generator
def build_generator():
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(
        Dense(X_train.shape[1], activation="linear")
    )  # Output same shape as input features
    return model


# Discriminator
def build_discriminator():
    model = Sequential()
    model.add(Dense(128, input_dim=X_train.shape[1], activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))  # Output is a single probability
    return model


# Compile GAN Model
def compile_gan(generator, discriminator):
    discriminator.compile(
        loss="binary_crossentropy", optimizer=Adam(0.0002, 0.5), metrics=["accuracy"]
    )

    # Combined model (Generator and Discriminator)
    discriminator.trainable = False
    gan_input = Input(shape=(X_train.shape[1],))
    generated_data = generator(gan_input)
    gan_output = discriminator(generated_data)

    gan = Model(gan_input, gan_output)
    gan.compile(loss="binary_crossentropy", optimizer=Adam(0.0002, 0.5))
    return gan


# Build and compile the models
generator = build_generator()
discriminator = build_discriminator()
gan = compile_gan(generator, discriminator)

# Training the GAN
epochs = 100
batch_size = 64
half_batch = batch_size // 2

for epoch in range(epochs):
    # Train Discriminator

    # Select a random half batch of real data
    idx = np.random.randint(0, X_train.shape[0], half_batch)
    real_data = X_train[idx]

    # Generate a half batch of fake data
    noise = np.random.normal(0, 1, (half_batch, X_train.shape[1]))
    fake_data = generator.predict(noise)

    # Combine real and fake data
    X_combined = np.concatenate([real_data, fake_data])
    y_combined = np.concatenate(
        [np.ones((half_batch, 1)), np.zeros((half_batch, 1))]
    )  # Labels: real=1, fake=0

    # Train the discriminator (real data = 1, fake data = 0)
    d_loss_real = discriminator.train_on_batch(real_data, np.ones((half_batch, 1)))
    d_loss_fake = discriminator.train_on_batch(fake_data, np.zeros((half_batch, 1)))
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train Generator
    noise = np.random.normal(0, 1, (batch_size, X_train.shape[1]))
    valid_y = np.ones(
        (batch_size, 1)
    )  # We want the generator to fool the discriminator into thinking these are real

    g_loss = gan.train_on_batch(noise, valid_y)

    # Print progress every 1000 epochs
    if epoch % 1000 == 0:
        print(
            f"{epoch} [D loss: {d_loss[0]}, acc.: {100 * d_loss[1]}%] [G loss: {g_loss}]"
        )

# Generate new synthetic loan data using the trained generator
noise = np.random.normal(0, 1, (X_test.shape[0], X_test.shape[1]))
synthetic_data = generator.predict(noise)

# Predict using discriminator
y_pred_gan = discriminator.predict(synthetic_data)

# Inverse scale the predicted values to compare to the original scale
y_pred_gan_inverse_scaled = y_scaler.inverse_transform(y_pred_gan).flatten()

# Predict loan amount for custom input
custom_input = pd.DataFrame(
    {
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

# Standardize the custom input
X_custom = scaler.transform(custom_input.drop(columns=["loan_amount"]))

# Generate synthetic predictions for custom input using the generator
y_custom_pred_gan = generator.predict(X_custom)

# Inverse transform to get the original loan amount scale
y_custom_pred_gan_inverse_scaled = y_scaler.inverse_transform(
    y_custom_pred_gan
).flatten()

print(f"\n\nPredicted loan amounts by GAN: \n{y_custom_pred_gan}")
print(f'\nActual applied loan amounts: \n{custom_input["loan_amount"]}')

# Classify based on the GAN predicted values
for i in range(len(y_custom_pred_gan_inverse_scaled)):
    predicted_loan = y_custom_pred_gan_inverse_scaled[i]
    actual_loan = custom_input["loan_amount"].values[i]

    if predicted_loan >= actual_loan:
        print(
            colored(
                f"Test Case {i+1}: Loan will be approved (Predicted: {predicted_loan}, Applied: {actual_loan})",
                "green",
            )
        )
    else:
        print(
            colored(
                f"Test Case {i+1}: Loan will not be approved (Predicted: {predicted_loan}, Applied: {actual_loan})",
                "red",
            )
        )
