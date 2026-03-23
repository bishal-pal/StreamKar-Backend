import torch

COLLECTION_NAME = "faq_collection"
BATCH_SIZE = 100

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
