import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from data.gan2_dataset import Gan2Dataset
import numpy as np


class DummyOpt:
    def __init__(self):
        self.dataroot = "/home/mds-user/Documents/Masterthesis/Data/train_gan2_data/msseg/preprocessing_results"
        self.phase = "train"  # oder "test"

# === 3. Dataset erstellen ===
opt = DummyOpt()
dataset = Gan2Dataset(opt)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# === 4. Ein Batch anschauen ===
batch = next(iter(loader))
A = batch['A']  # masked flair
B = batch['B']  # original flair
mask = batch['mask']

print("A shape:", A.shape)
print("B shape:", B.shape)
print("Mask shape:", mask.shape)

print("labels mask: ", np.unique(mask))
print("labels A: ", np.unique(A))
print("labels B: ", np.unique(B))

def show_tensor(tensor, title, subplot_index):
    img = tensor.squeeze().numpy()
    plt.subplot(1, 3, subplot_index)
    plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.axis("off")

plt.figure(figsize=(12, 4))
show_tensor(A[0], "Masked Flair (A)", 1)
show_tensor(B[0], "Original Flair (B)", 2)
show_tensor(mask[0], "Lesion Mask", 3)
plt.tight_layout()
plt.show()
