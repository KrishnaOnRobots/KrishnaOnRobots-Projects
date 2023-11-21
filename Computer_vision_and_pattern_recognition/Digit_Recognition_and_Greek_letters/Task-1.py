# Gopisainath & Krishna Prasath Senthil Kumaran

# import statements
import torch
import torchvision
import matplotlib.pyplot as plt
import sys
import torch.nn as nn
import torch.nn.functional as functional

batch_size_train = 64
batch_size_test = 1000
n_epochs = 5
train_losses = []
train_counter = []
test_losses = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# class definitions
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, device="cuda")
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, device="cuda")
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50, device="cuda")
        self.fc2 = nn.Linear(50, 10, device="cuda")

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
def mnist(test_loader):
    examples = enumerate(test_loader)
    # examples = examples.to(device)
    batch_idx, (example_data, example_targets) = next(examples)
    # print(example_data.shape)

    plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()


def train(epoch, network, train_loader, optimizer):
    log_interval = 10

    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = network(data)
        loss = nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))


def test(network, test_loader):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = network(data)
            test_loss += nn.functional.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


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

    test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

    learning_rate = 0.01
    momentum = 0.5
    torch.backends.cudnn.enabled = False
    torch.manual_seed(42)

    mnist(test_loader)
    network = Net()
    network.to(device)
    optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
    test(network, test_loader)
    for epoch in range(1, n_epochs + 1):
        train(epoch, network, train_loader, optimizer)
        test(network, test_loader)
    plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()

    filename = 'model.sav'
    #     pickle.dump(network, open(filename, 'wb'))
    torch.save(network, filename)

    return 0


if __name__ == "__main__":
    main(sys.argv)
