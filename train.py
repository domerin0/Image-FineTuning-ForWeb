import torch
import torch.nn as nn
import torch.optim as optim
import math
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights
import time
import random
import copy

# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
data_dir = "./data/data"

# Training parameters
num_classes = 7
batch_size = 16
num_epochs = 2
input_size = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# How much of the dataset to use for training
train_frac = 0.8


def train(
    model, dataloaders, criterion, optimizer, num_epochs=25
):
    start = time.time()
    test_accuracy_hist = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("Epoch {0}/{1}".format(epoch, num_epochs))
        print('-' * 10)

        # train iteration
        iterate_dataset(
            model, dataloaders, criterion,
            optimizer, test_accuracy_hist,
            best_acc=0.0, mode="train"
        )

        # test iteration
        best_model_wts, best_acc = iterate_dataset(
            model, dataloaders, criterion,
            optimizer, test_accuracy_hist,
            best_acc, mode="test"
        )

    time_elapsed = time.time() - start
    print(
        'Completed in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60)
    )
    print('Best test accuracy: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, test_accuracy_hist


def iterate_dataset(model, dataloaders, criterion, optimizer, test_accuracy_hist, best_acc=0, mode="train"):
    best_model_wts = None

    epoch_start = time.time()
    if mode == 'train':
        # Set model to training mode
        model.train()
    else:
        # Set model to evaluate mode
        model.eval()

    running_loss = 0.0
    running_corrects = 0
    # Now iterate over test set
    for inputs, labels in dataloaders[mode]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(mode == "train"):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            # backward + optimize only if in training phase
            if mode == 'train':
                loss.backward()
                optimizer.step()
        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloaders[mode].dataset)
    epoch_acc = running_corrects.double(
    ) / len(dataloaders[mode].dataset)

    print('{} Loss: {:.4f} Acc: {:.4f} Dur: {:.0f}s'.format(
        mode, epoch_loss, epoch_acc, (time.time() - epoch_start) % 60)
    )

    # deep copy the model
    if mode != "train" and epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())
    if mode != "train":
        test_accuracy_hist.append(epoch_acc)
    return best_model_wts, best_acc


def get_data():
    data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image_dataset = datasets.ImageFolder(data_dir, transform=data_transforms)

    dataset_size = len(image_dataset.imgs)
    print("{0} images in dataset".format(dataset_size))
    indices = list(range(dataset_size))
    split = int(math.floor(train_frac * dataset_size))
    random.shuffle(indices)

    test_indices, train_indices = indices[split:], indices[:split]

    print("{0} images in training dataset".format(len(train_indices)))
    print("{0} images in test dataset".format(len(test_indices)))

    test_set = torch.utils.data.Subset(
        image_dataset, test_indices,
    )

    train_set = torch.utils.data.Subset(
        image_dataset, train_indices,
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size
    )

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size,
    )

    dataloaders_dict = {
        "train": train_loader,
        "test": test_loader
    }

    return dataloaders_dict


def get_model(num_classes):
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    # Get input size to last layer
    num_ftrs = model.fc.in_features

    # Replace last layer with trainable linear layer with 7 classes
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model


def main():
    model = get_model(num_classes)

    # Send the model to the right device
    model = model.to(device)

    params_to_update = model.parameters()
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)

    dataloaders = get_data()

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    # Train and evaluate
    model, _ = train(
        model, dataloaders, criterion,
        optimizer, num_epochs=num_epochs
    )

    torch.save(model, "best.pt")


if __name__ == '__main__':
    main()
