from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes

# Step 1: Generate ECDSA keys
private_key = ec.generate_private_key(ec.SECP256R1(), default_backend())
public_key = private_key.public_key()

# Step 2: Define the message to be signed
data = b"This is a secret message."

# Step 3: Sign the message
signature = private_key.sign(data, ec.ECDSA(hashes.SHA256()))
print("ECDSA Signature:", signature.hex())

# Step 4: Verify the signature
try:
    public_key.verify(signature, data, ec.ECDSA(hashes.SHA256()))
    print("Signature is valid!")
except Exception as e:
    print("Signature is invalid:", str(e))
