# Import necessary libraries
import torch
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler

# Define the path to the directory containing the image dataset
path = "Plastics Classification"

# Define a set of image transformations to be applied to the dataset
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Resize((300, 300), antialias=True),
                                            torchvision.transforms.RandomHorizontalFlip(),
                                            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load the image dataset from the specified directory path and apply the set of transformations
dataset = torchvision.datasets.ImageFolder(root=path, transform=transform)

# Divide the dataset into training and testing subsets
train_indices = range(0, int(len(dataset)*0.8))
test_indices = range(int(len(dataset)*0.8), len(dataset))

# Create sampler objects that will be used to randomly sample the training and testing subsets
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

# Create DataLoader objects that will be used to load the training and testing subsets into batches
train_loader = torch.utils.data.DataLoader(dataset, batch_size = 64, sampler = train_sampler)
test_loader = torch.utils.data.DataLoader(dataset, batch_size = 64, sampler = test_sampler)
