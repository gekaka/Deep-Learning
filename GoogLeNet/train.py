
import torch.nn as nn
from torchvision import transforms, datasets
import torchvision
import json
import os
import torch.optim as optim
import torch
import time
from model import GoogLeNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224), 
                                 transforms.RandomHorizontalFlip(), 
                                 transforms.ToTensor(), 
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]), 
    "val": transforms.Compose([transforms.Resize((224, 224)), 
                               transforms.ToTensor(), 
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
data_root = os.path.abspath(os.path.join(os.getcwd(), "."))  # get data root path
image_path = data_root + "/data_set/flower_data/"  # flower data set path

train_dataset = datasets.ImageFolder(root=image_path+"train", 
                                     transform=data_transform["train"])
train_num = len(train_dataset)

# {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
flower_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in flower_list.items())
# write dict into json file
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset, 
                                           batch_size=batch_size, shuffle=True, 
                                           num_workers=0)

validate_dataset = datasets.ImageFolder(root=image_path + "val", 
                                        transform=data_transform["val"])
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset, 
                                              batch_size=batch_size, shuffle=False, 
                                              num_workers=0)

# test_data_iter = iter(validate_loader)
# test_image, test_label = test_data_iter.next()

# net = torchvision.models.googlenet(num_classes=5)
# model_dict = net.state_dict()
# pretrain_model = torch.load("googlenet.pth")
# del_list = ["aux1.fc2.weight", "aux1.fc2.bias", 
#             "aux2.fc2.weight", "aux2.fc2.bias", 
#             "fc.weight", "fc.bias"]
# pretrain_dict = {k: v for k, v in pretrain_model.item() if k not in del_list}
# model_dict.update(pretrain_dict)
# net.load_state_dict(model_dict)
net = GoogLeNet(num_classes=5, aux_logits=True, init_weights=True)
net.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0003)

best_acc = 0.0
save_path = './googleNet.pth'
for epoch in range(30):
    # train
    net.train()
    running_loss = 0.0
    t1 = time.perf_counter()
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()
        logits, aux_logits2, aux_logits1 = net(images.to(device))
        loss0 = loss_function(logits, labels.to(device))
        loss1 = loss_function(aux_logits1, labels.to(device))
        loss2 = loss_function(aux_logits2, labels.to(device))
        loss = loss0 + loss1 * 0.3 + loss2 * 0.3
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # print train process
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain_loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
    print()
    print(time.perf_counter()-t1)

    # validate
    net.eval()
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
