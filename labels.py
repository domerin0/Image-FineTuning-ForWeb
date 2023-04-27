import matplotlib.pyplot as plt
from torchvision import datasets
import torch

labels_map = {
    0: "bike",
    1: "car",
    2: "cat",
    3: "dog",
    4: "flower",
    5: "horse",
    6: "human",
}

data_dir = "./data/data"


def visualize():
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 5, 5
    dataset = datasets.ImageFolder(data_dir)
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        img, label = dataset[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(img, cmap="gray")
    plt.show()


if __name__ == '__main__':
    visualize()
