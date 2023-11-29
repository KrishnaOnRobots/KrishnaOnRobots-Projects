import numpy as np
import torch
import torchvision
import torch.nn as nn
import random
import matplotlib.pyplot as plt

train_path = "asl_alphabet_train"
test_path = "asl_alphabet_test"

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Resize((300, 300), antialias=True),
                                            torchvision.transforms.RandomHorizontalFlip(),
                                            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=transform)
test_dataset = torchvision.datasets.ImageFolder(root=test_path, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 200, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 64, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model = torchvision.models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 29)

# print(network)
model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr= 0.001)

# Train the model
num_epochs = 5
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
        # inputs = inputs
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
        filename = 'sign_lang_model.pth'
        torch.save(model.state_dict(), filename)

filename = 'sign_lang_model.pth'
state = torch.load(filename)
model.load_state_dict(state)

dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z',26: 'nothing', 27: 'space',28: 'del'}

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
columns = 6
rows = 5
for i in range(28):
    fig.add_subplot(rows, columns, i+1)
    image = images[i].cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = image * 0.5 + 0.5
    plt.imshow(image)
    plt.title(f"Predicted: {dict[predicted[i].item()]}\nActual: {dict[labels[i].item()]}")
    plt.axis('off')

plt.show()