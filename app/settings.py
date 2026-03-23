import torch

COLLECTION_NAME = "streamkar"
BATCH_SIZE = 100

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
