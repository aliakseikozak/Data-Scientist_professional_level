import torch

class Config:
    DEVICE = torch.device('cpu')  # CPU на твоём устройстве
    BATCH_SIZE = 16
    HIDDEN_SIZES = [128, 64, 32]
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 500
    CONFIDENCE_THRESHOLD = 0.7
