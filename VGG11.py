
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Hyper-parameters 
NUM_EPOCHS = 6
BATCH_SIZE = 30
LERNING_RATE = 0.001
DATSET_SIZE = 6000
PATH = './cnn.pth'

#Test for Generalization
HORIZONTAL_FLIP = 0
VERTICAL_FLIP = 0
GAUSSIAN_NOISE_001 = 0
GAUSSIAN_NOISE_01 = 0
GAUSSIAN_NOISE_1 = 0

common_transformations = [
    transforms.Resize((32, 32)),  # Resize images from 28x28 to 32x32
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize the images
]

title_prefix = ""
test_transformations = []
if HORIZONTAL_FLIP:
    test_transformations.append(transforms.RandomHorizontalFlip(p=1))
    title_prefix = "horizontal flip - "
elif VERTICAL_FLIP:
    test_transformations.append(transforms.RandomVerticalFlip(p=1))
    title_prefix = "vertical flip - "
elif GAUSSIAN_NOISE_001:
    test_transformations.append(transforms.Lambda(lambda x: x + 0.01 * torch.randn_like(x)))
    title_prefix = "gaussian noise (var=0.01) - "
elif GAUSSIAN_NOISE_01:
    test_transformations.append(transforms.Lambda(lambda x: x + 0.1 * torch.randn_like(x)))
    title_prefix = "gaussian noise (var=0.1) - "
elif GAUSSIAN_NOISE_1:
    test_transformations.append(transforms.Lambda(lambda x: x + 1 * torch.randn_like(x)))
    title_prefix = "gaussian noise (var=1) - "

random_choice_transform = transforms.RandomChoice([
    transforms.RandomHorizontalFlip(p=1),
    transforms.RandomVerticalFlip(p=1),
    transforms.Lambda(lambda x: x + 0.01 * torch.randn_like(x)),
    transforms.Lambda(lambda x: x + 0.1 * torch.randn_like(x)),
    transforms.Lambda(lambda x: x + 1 * torch.randn_like(x)),
    transforms.Lambda(lambda x: x)])
    

train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    #random_choice_transform, 
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize the images
])

# Compose transformations
#train_transform = transforms.Compose(common_transformations + random_choice_transform)
test_transform = transforms.Compose(common_transformations + test_transformations)

# Load the MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
# Reduce dataset size
train_dataset.data = train_dataset.data[:DATSET_SIZE]
train_dataset.targets = train_dataset.targets[:DATSET_SIZE]

# Create DataLoader with the reduced dataset
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
                                          


# PRINT IMAGES - For curiosty 
# Function to denormalize and display images
def imshow(img):
    img = img * 0.3081 + 0.1307  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    plt.show()

# Get a batch of training data
#dataiter = iter(train_loader)
#images, labels = next(dataiter)

# Show images
#imshow(torchvision.utils.make_grid(images[:5]))

# Print labels
#print(' '.join(f'{labels[j].item()}' for j in range(5)))

###################### CNN ######################

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




classes = ('0','1', '2', '3', '4',
           '5', '6', '7', '8', '9')

# get some random training images
dataiter = iter(train_loader)
images, labels = next(dataiter)

# show images
#imshow(torchvision.utils.make_grid(images))

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(4096, 10)
        )

    def forward(self, x):    
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x


def train_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.train()
    train_loss, correct = 0, 0

    for batch, (images, labels) in enumerate(train_loader): 
        images = images.to(device)
        labels = labels.to(device)

        #Forward Pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        #Compute Loss
        train_loss += loss.item()

        #max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()

        # Backpropagation and Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_train_loss = train_loss / num_batches
    train_accuracy = correct / size
    return train_accuracy, average_train_loss

def test_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for batch, (images,labels) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            loss = loss_fn(outputs, labels)
            test_loss += loss.item()

            #max returns (value ,index)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

    epoch_average_test_loss = test_loss / num_batches
    epoch_test_accuracy = correct / size
    return epoch_test_accuracy, epoch_average_test_loss

def train(model):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LERNING_RATE)

    average_train_loss = []
    all_train_accuracy = []
    average_test_loss = []
    all_test_accuracy = []

    for epoch in range(NUM_EPOCHS):
        train_accuracy, train_loss = train_epoch(train_loader, model, criterion, optimizer)
        all_train_accuracy += [train_accuracy]
        average_train_loss += [train_loss]
        test_accuracy, test_loss = test_epoch(test_loader, model, criterion, optimizer)
        all_test_accuracy += [test_accuracy]
        average_test_loss += [test_loss]
        
        print(f'Epoch #{epoch+1}: \t train accuracy {train_accuracy:.3f}\t train loss {train_loss:.3f}\t test accuracy {test_accuracy:.3f}\t test loss {test_loss:.3f}')

    return all_train_accuracy, average_train_loss, all_test_accuracy, average_test_loss

def test_saved_model(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LERNING_RATE)
    test_accuracy, test_loss = test_epoch(test_loader, model, criterion, optimizer)
    return test_accuracy



model = ConvNet().to(device) 
#RETRAIN
all_train_accuracy, average_train_loss, all_test_accuracy, average_test_loss = train(model)
print('Finished Training')
torch.save(model.state_dict(), PATH)

plots = [(all_train_accuracy, 'train accuracy'),
         (average_train_loss, 'train loss'),
         (all_test_accuracy,'test accuracy'),
         (average_test_loss,'test loss')]

for plot in plots:
    plt.plot(plot[0])
    plt.title(title_prefix + plot[1] + ' vs epoch')
    plt.ylabel(plot[1])
    plt.xlabel('epoch')
    plt.xticks(range(0, NUM_EPOCHS, 1))
    plt.show()

#TEST SAVED MODEL
# model.load_state_dict(torch.load(PATH))
# test_accuracy = test_saved_model(model)
# print(title_prefix + ' test accuracy: '+ str(test_accuracy))




