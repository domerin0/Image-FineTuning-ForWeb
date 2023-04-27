import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms
import numpy as np
from labels import labels_map

data_dir = "./data/data"

# The image input size we need
input_size = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_transforms = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def predict_image(model, image):
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = image_tensor
    input = input.to(device)
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    print(list(output.data.cpu()[0]))
    print(index)
    return index


def get_random_images(num):
    data = datasets.ImageFolder(data_dir, transform=test_transforms)
    indices = list(range(len(data)))
    np.random.shuffle(indices)
    idx = indices[:num]
    from torch.utils.data.sampler import SubsetRandomSampler
    sampler = SubsetRandomSampler(idx)
    loader = torch.utils.data.DataLoader(
        data,
        sampler=sampler,
        batch_size=num)
    dataiter = iter(loader)
    images, labels = next(dataiter)
    return images, labels


def main():
    model = torch.load('./best.pt')
    model.eval()

    to_pil = transforms.ToPILImage()
    images, labels = get_random_images(5)
    fig = plt.figure(figsize=(10, 10))
    for ii in range(len(images)):
        image = to_pil(images[ii])
        index = predict_image(model, image)
        label = labels_map[index]
        sub = fig.add_subplot(1, len(images), ii+1)
        sub.set_title("Actual: {0}{1}\nGuess: {2}{3}".format(
            labels_map[int(labels[ii])], int(labels[ii]), label, index))
        plt.axis('off')
        plt.imshow(image, cmap="gray")
    plt.show()


if __name__ == '__main__':
    main()
