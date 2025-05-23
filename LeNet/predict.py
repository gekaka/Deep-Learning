
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import LeNet

transform = transforms.Compose(
    [transforms.Resize((32, 32)), 
     transforms.ToTensor(), 
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

classes = ('plane', 'car', 'bird', 'cat', 
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = LeNet()
net.load_state_dict(torch.load('LeNet.pth'))

im = Image.open('plane.webp')
im = transform(im)  # [C, H, W]
im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]

with torch.no_grad():
    outputs = net(im)
    predict = torch.max(outputs, dim=1)[1].numpy()
print(classes[int(predict)])
