import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models

# Load the MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)

# Convert grayscale images to RGB images
grayscale_to_rgb = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.expand(3, -1, -1)),
    transforms.Normalize((0.1307,), (0.3081,))
])


train_dataset.transform = grayscale_to_rgb

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize the ResNet18 model
model = models.resnet18(pretrained=True)

# Replace the last fully connected layer with a new one for digit recognition
model.fc = nn.Linear(model.fc.in_features, 10)

# Set the device to GPU if available, else use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(10):
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        # Move the images and labels to the GPU if available
        images = images.to(device)
        labels = labels.to(device)
        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        # Print statistics
        running_loss += loss.item()
        if (i+1) % 100 == 0:
            print('[Epoch %d, Batch %d] Loss: %.3f' % (epoch+1, i+1, running_loss/100))
            running_loss = 0.0

# Save the trained model parameters
# torch.save(model.state_dict(), 'resnet18_mnist.pt')
filename = 'resnet18_mnist.sav'
#     pickle.dump(network, open(filename, 'wb'))
torch.save(model, filename)