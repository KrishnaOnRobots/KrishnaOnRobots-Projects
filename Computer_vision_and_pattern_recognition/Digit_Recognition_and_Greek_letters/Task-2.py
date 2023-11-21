# Gopisainath

# import statements
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as functional
import matplotlib.pyplot as plt 
import cv2 as cv
import sys

batch_size_train = 64
batch_size_test = 1000
n_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class definitions
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    # computes a forward pass for the network
    # methods need a summary comment
    def forward(self, x):
        x = functional.relu(functional.max_pool2d(self.conv1(x), 2))
        x = functional.relu(functional.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = functional.relu(self.fc1(x))
        x = functional.dropout(x, training=self.training)
        x = self.fc2(x)
        return functional.log_softmax(x)


# useful functions with a comment for each function
def layers_plot(weights):
    plt.figure()
    for i in range(10):
        plt.subplot(3,4,i+1)
        plt.tight_layout()
        # print(weights.shape)
        plt.imshow(weights[i][0], interpolation='none')
        plt.title(f"Filter: {i}")
        plt.xticks([])
        plt.yticks([])
    plt.show()
    return

def viz(network, example_data, weights):
    with torch.no_grad():
        output = network(example_data[0])
        img = example_data[0].numpy()
    plt.figure()
    for i in range(0,19,2):
        plt.subplot(5,4,i+1)
        plt.tight_layout()
        a = i/2
        plt.imshow(weights[int(a)][0], cmap="gray",interpolation='none')
        plt.subplot(5,4,i+2)
        img = cv.filter2D(src=img.reshape(28,28,1), ddepth=-1, kernel=weights[int(a)][0])
        plt.imshow(img, cmap="gray",interpolation='none')
        plt.xticks([])
        plt.yticks([])
    plt.show()
    return

# main function (yes, it needs a comment too)
def main(argv):
    # handle any command line arguments in argv

    # main function code

    train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/files/', train=True, download=True,
                            transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                (0.1307,), (0.3081,))
                            ])),
    batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/files/', train=False, download=True,
                            transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                (0.1307,), (0.3081,))
                            ])),
    batch_size=batch_size_test, shuffle=True)

    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    network = Net()
    filename = 'model.sav'
    network = torch.load(filename, map_location=torch.device('cpu'))
    network.eval()
    print(network)

    weights = network.conv1.weight

    weights = weights.detach().numpy()

    layers_plot(weights)
    viz(network, example_data, weights)

    return

if __name__ == "__main__":
    main(sys.argv)