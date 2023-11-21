# Gopisainath

# import statements
import sys
import torch 
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F

batch_size_train = [64, 128]
batch_size_test = 1000
n_epochs = [3, 5]
learning_rate = 0.001
momentum = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class definitions
class Net(nn.Module):
    
    def __init__(self, num_conv_layers, kernal_size):
        super(Net, self).__init__()
        self.img_size = 0
        self.in_channels = 1
        self.row_size = 28
        self.conv = []
        for n in range(num_conv_layers):
            self.out_channels = (n+1)*10
            self.conv.append(nn.Conv2d(self.in_channels,self.out_channels, kernel_size= kernal_size,device=device))
            # print(self.conv[0])
            self.in_channels = self.out_channels
            self.row_size = int(((abs(self.row_size-kernal_size))+1)/2)
            self.img_size = self.row_size*self.row_size*self.out_channels

        self.drop = nn.Dropout2d()
        self.fc1 = nn.Linear(self.img_size, 60)
        self.fc2 = nn.Linear(60, 10)

    # computes a forward pass for the network
    # methods need a summary comment
    def forward(self, x, num_conv_layers, activation_func):
        
        # print(f"self = {self.conv[0]}")
        for n in range(num_conv_layers):
            if activation_func == 1: 
                x = F.relu(F.max_pool2d(self.drop(self.conv[n](x)),2))
            elif activation_func == 2:
                x = F.gelu(F.max_pool2d(self.drop(self.conv[n](x)),2))
            elif activation_func == 3:
                x = F.sigmoid(F.max_pool2d(self.drop(self.conv[n](x)),2))
        
        x = x.view(-1, self.img_size)
        if activation_func == 1:
            x = F.relu(self.fc1(x))
        elif activation_func == 2:
            x = F.gelu(self.fc1(x))
        elif activation_func == 3:
            x = F.sigmoid(self.fc1(x))
        
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

# useful functions with a comment for each function
def train(epoch, layers, func, network, train_loader, optimizer, train_losses, train_counter):
    network.train()
    for ids, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = network(data, layers, func)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        if ids % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, ids * len(data), len(train_loader.dataset),100. * ids / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (ids*64) + ((epoch-1)*len(train_loader.dataset)))

    return

def test(layers, func, network, test_loader, test_losses):
    network.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
          data = data.to(device)
          target = target.to(device)
          output = network(data, layers, func)
          test_loss += nn.functional.nll_loss(output, target, size_average=False).item()
          pred = output.data.max(1, keepdim=True)[1]
          correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
      test_loss, correct, len(test_loader.dataset),
      100. * correct / len(test_loader.dataset)))

    return

def plot_curve(train_counter, train_losses, test_counter, test_losses, filename):

    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.savefig(filename)
    plt.close()

# main function (yes, it needs a comment too)
def main(argv):
    # handle any command line arguments in argv

    # main function code
    kernal_size = torch.tensor([2, 3, 4, 5], dtype=torch.int8, device=device)       #2,3,4,5
    num_conv_layers = torch.tensor([1, 2], dtype=torch.int8, device=device)         #1,2
    dict = {1:"relu", 2:"gelu", 3:"sigmoid"}                                        
    activation_func = torch.tensor([3], dtype=torch.int8, device=device)      #1,2,3

    for size in kernal_size:
        for layers in num_conv_layers:
            for func in activation_func:
                for batch in batch_size_train:
                    for n_epoch in n_epochs:

                        network = Net(layers.item(), size.item())
                        network.to(device)
                        optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
                        print('______________________________')
                        print(f'Number of Epochs: {n_epoch}')
                        print(f'Train Batch Size: {batch}')
                        print(f'Number of Convolution Layer: {layers}')
                        print(f'kernel Size: {size}')
                        print(f'activation_function: {func}')
                        print('______________________________')

                        train_loader = torch.utils.data.DataLoader(
                            torchvision.datasets.FashionMNIST('/files/', train=True, download=True,
                              transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                  (0.1307,), (0.3081,))
                              ])),
                            batch_size=batch, shuffle=True)

                        test_loader = torch.utils.data.DataLoader(
                            torchvision.datasets.FashionMNIST('/files/', train=False, download=True,
                              transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                  (0.1307,), (0.3081,))
                              ])),
                            batch_size=batch_size_test, shuffle=True)
                        
                        filename = f'plots/{n_epoch}_{batch}_{layers.item()}_{size.item()}_{dict[func.item()]}.png'
                        train_losses = []
                        train_counter = []
                        test_losses = []
                        test_counter = [i * len(train_loader.dataset) for i in range(n_epoch)]

                        for epoch in range(1, n_epoch + 1):  
                            train(epoch, layers.item(), func.item(), network, train_loader, optimizer, train_losses, train_counter)
                            test(layers.item(), func.item(), network, test_loader, test_losses)
                        # print(test_counter)
                        plot_curve(train_counter, train_losses, test_counter, test_losses, filename)

    return

if __name__ == "__main__":
    main(sys.argv)
