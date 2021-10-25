##0 Example - Transfer Attack.py

from torch.utils.data import DataLoader, TensorDataset
import torchvision.utils
import torchvision.datasets as dsets
import warnings
warnings.filterwarnings('ignore')
from models import Source, Target
from DIFGSM import *


##1 check version
print("PyTorch", torch.__version__)
print("Torchvision", torchvision.__version__)


##2 Load Data
batch_size = 24

cifar10_train = dsets.CIFAR10(root='./data', train=True,
                              download=True, transform=transforms.ToTensor())
cifar10_test  = dsets.CIFAR10(root='./data', train=False,
                              download=True, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(cifar10_train,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(cifar10_test,
                                          batch_size=batch_size,
                                          shuffle=False)


##3 Attack Source Model & Save Adversarial Images
model = Source()
model.load_state_dict(torch.load("./data/source.pth"))
model = model.eval().cuda()

atk = DIFGSM(model, eps=16 / 255, alpha=2 / 255, steps=10, di_pad_amount=5)
atk.set_return_type('int') # Save as integer.
print('\n#################Source Model#################')
atk.save(data_loader=test_loader, save_path="./data/cifar10_DIFGSM.pt", verbose=True)


##4 Load Adversarial Images & Attack Target Model
adv_images, adv_labels = torch.load("./data/cifar10_DIFGSM.pt")
adv_data = TensorDataset(adv_images.float()/255, adv_labels)
adv_loader = DataLoader(adv_data, batch_size=128, shuffle=False)

model = Target().cuda()
model.load_state_dict(torch.load("./data/target.pth"))


##5 Target Model : Clean Accuracy
print('#################Target Model#################')
model.eval()
correct = 0
total = 0

for images, labels in test_loader:
    images = images.cuda()
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels.cuda()).sum()

print('Standard accuracy: %.2f %%' % (100 * float(correct) / total))


##6 Target Model : Robust Accuracy
model.eval()
correct = 0
total = 0

for images, labels in adv_loader:
    images = images.cuda()
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels.cuda()).sum()

print('Robust accuracy: %.2f %%' % (100 * float(correct) / total))
