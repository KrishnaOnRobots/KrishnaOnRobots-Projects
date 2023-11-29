# Import necessary libraries
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set path and transform
path = "Plastics Classification"
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((300, 300), antialias=True),
                                transforms.RandomHorizontalFlip(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load data using sampler
dataset = torchvision.datasets.ImageFolder(root=path, transform=transform)

train_indices = range(0, int(len(dataset)*0.8))
test_indices = range(int(len(dataset)*0.8), len(dataset))

train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=64, sampler=test_sampler)

# Define VGGNet model
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)
print(model)

# Send model to device
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Train the model
num_epochs = 10
train_acc = []
test_acc = []

best_accuracy = 0.0
for epoch in range(num_epochs):
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    correct_test = 0
    total_test = 0

    # Train
    model.train()
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        if (i+1) % 10 == 0:
            print('[Epoch %d, Batch %d] Loss: %.3f' % (epoch+1, i+1, running_loss/10))
            running_loss = 0.0        

    # Test
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)

            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    # Calculate and store accuracy
    train_accuracy = 100 * correct_train / total_train
    test_accuracy = 100 * correct_test / total_test
    train_acc.append(train_accuracy)
    test_acc.append(test_accuracy)

    print('Epoch [{}/{}], Loss: {:.4f}, Train Accuracy: {:.2f}%, Test Accuracy: {:.2f}%'
          .format(epoch+1, num_epochs, running_loss/len(train_loader), train_accuracy, test_accuracy))

    # Save the model if it has the best accuracy on the test set so far
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        filename = 'resnet18_10_128.pth'
        torch.save(model.state_dict(), filename)
    

# filename = 'resnet18_plastics_classification.pth'
# state = torch.load(filename)
# model.load_state_dict(state)

# Plot accuracy graph
plt.plot(range(num_epochs), train_acc, 'b', label='Training Accuracy')
plt.plot(range(num_epochs), test_acc, 'r', label='Testing Accuracy')
plt.title('Training and Testing Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.show()

# Visualize some random images
examples = enumerate(test_loader)
batch_idx, (images, labels) = next(examples)

# Send images and labels to device
images = images.to(device)
labels = labels.to(device)

# Predict labels for the images
outputs = model(images)
_, predicted = torch.max(outputs, 1)

# Prepare image grid
fig = plt.figure(figsize=(10, 10))
columns = 5
rows = 2
for i in range(columns*rows):
    fig.add_subplot(rows, columns, i+1)
    image = images[i].cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = image * 0.5 + 0.5
    plt.imshow(image)
    plt.title('Predicted: {}\nActual: {}'.format(predicted[i].item(), labels[i].item()))
    plt.axis('off')
plt.show()