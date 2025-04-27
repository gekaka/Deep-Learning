
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import json
import matplotlib.pyplot as plt
import os
import torch.optim as optim
from model import resnet34, resnet101
import torchvision.models.resnet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

data_transfrom = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224), 
                                 transforms.RandomHorizontalFlip(), 
                                 transforms.ToTensor(), 
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]), 
    "val": transforms.Compose([transforms.Resize(256),  # 将原图形的长宽比不变, 将最短边长缩放到256, 详见Resize的函数说明: sequence和int的区别
                               transforms.CenterCrop(224),  # 使用中心裁剪来裁剪一个 224 x 224 大小的图片
                               transforms.ToTensor(), 
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}


data_root = os.path.abspath(os.path.join(os.getcwd(), "."))  # get data root path
image_path = data_root + "/data_set/flower_data/"  # flower data set path

train_dataset = datasets.ImageFolder(root=image_path+"train", 
                                     transform=data_transfrom["train"])

train_num = len(train_dataset)

# {'daisy':0, 'dandlion':1, 'roses':2, 'sunflower':3, 'tulips':4}
flower_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in flower_list.items())
# write dict into json file
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

batch_size = 16
train_loader = torch.utils.data.DataLoader(train_dataset, 
                                           batch_size=batch_size, shuffle=True, 
                                           num_workers=0)  # 当使用Linux系统时, 将num_workers设为大于0的数能加速图像预处理的过程

validate_dataset = datasets.ImageFolder(root=image_path + 'val', 
                                        transform=data_transfrom["val"])
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset, 
                                              batch_size=batch_size, shuffle=False, 
                                              num_workers=0)  # num_workers的设值同上

net = resnet34()  # 未传入参数, 故num_classes默认=1000; 这里使用迁移学习的方法训练模型, 因此不设置参数
net.to(device)
# load pretrain weights
model_weight_path = "./resnet34-pre.pth"
missing_keys, unexpected_keys = net.load_state_dict(torch.load(model_weight_path), strict=False)
# for param in net.parameters():
#     param.requires_grad = False
# change fc layer structure
inchannel = net.fc.in_features
net.fc = nn.Linear(inchannel, 5)  # 对net中的fc进行重新定义, 但此过程默认在CPU中完成
net.to(device)  # 上一步使得已经转到GPU中的模型, 其中一部分又转回CPU, 因此这一步须将整个模型再次转回GPU

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)

best_acc = 0.0
save_path = './resNet34.pth'
for epoch in range(3):
    # train
    net.train()  # 这行代码一定要加, 因为Batch Normalization在训练和测试的过程中执行的方法是不一样的
    running_loss = 0.0
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()
        logits = net(images.to(device))
        loss = loss_function(logits, labels.to(device))
        loss.backward()
        optimizer.step()
    
        # print statistics
        running_loss += loss.item()
        # print train process
        rate = (step+1)/len(train_loader)
        a = "*" * int(rate * 50)
        b = "*" * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.4f}".format(int(rate*100), a, b, loss), end="")
    print()

    # validate
    net.eval()  # 这行代码一定要加, 因为Batch Normalization在训练和测试的过程中执行的方法是不一样的
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        for data_test in validate_loader:
            test_images, test_labels = data_test
            outputs = net(test_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == test_labels.to(device)).sum().item()
        accurate_test = acc / val_num
        if accurate_test > best_acc:
            best_acc = accurate_test
            torch.save(net.state_dict(), save_path)
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' % 
              (epoch + 1, running_loss / step, acc / val_num))

print('Finished Training')
