
import os
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

from model import resnext50_32x4d


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

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
    image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"), 
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
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                            batch_size=batch_size, shuffle=True, 
                                            num_workers=nw)  # 当使用Linux系统时, 将num_workers设为大于0的数能加速图像预处理的过程

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"), 
                                            transform=data_transfrom["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, 
                                                batch_size=batch_size, shuffle=False, 
                                                num_workers=nw)  # num_workers的设值同上

    print("using {} images for training, {} images fot validation.".format(train_num, 
                                                                        val_num))

    net = resnext50_32x4d()  # 未传入参数, 故num_classes默认=1000; 这里使用迁移学习的方法训练模型, 因此不设置参数
    # load pretrain weights
    model_weight_path = "./resnext50_32x4d.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    for param in net.parameters():
        param.requires_grad = False

    # change fc layer structure
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 5)  # 对net中的fc进行重新定义, 但此过程默认在CPU中完成
    net.to(device)  # 上一步使得已经转到GPU中的模型, 其中一部分又转回CPU, 因此这一步须将整个模型再次转回GPU

    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    epochs = 3
    best_acc = 0.0
    save_path = './resNext50.pth'
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()  # 这行代码一定要加, 因为Batch Normalization在训练和测试的过程中执行的方法是不一样的
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()
        
            # print statistics
            running_loss += loss.item()
            # print train process
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                        epochs,
                                                                        loss)

        # validate
        net.eval()  # 这行代码一定要加, 因为Batch Normalization在训练和测试的过程中执行的方法是不一样的
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                            epochs)
                
            val_accurate = acc / val_num
            print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
                (epoch + 1, running_loss / train_steps, val_accurate))
            
            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()
