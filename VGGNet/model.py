
import torch.nn as nn
import torch

class VGG(nn.Module):
    def __init__(self, features, class_num=1000, init_weights=False):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5), 
            nn.Linear(512*7*7, 2048),    # 原论文中本行的2048应为4096
            nn.ReLU(True), 
            nn.Dropout(p=0.5), 
            nn.Linear(2048, 2048),    # 原论文中本行的两个2048应均为4096
            nn.ReLU(True), 
            nn.Linear(2048, class_num)    # 原论文中本行的2048应为4096
        )
        if init_weights:
            self._initialize_weights()
    
    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.features(x)
        # N x 512 x 7 x 7
        x = torch.flatten(x, start_dim=1)
        # N x 512*7*7
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):    # 如果该层为卷积层
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)    # 用该方法初始化权重
                if m.bias is not None:    # 如果卷积核采用了偏置
                    nn.init.constant_(m.bias, 0)    # 就将偏置默认初始化为0
            elif isinstance(m, nn.Linear):    # 如果该层为全连接层
                nn.init.xavier_uniform_(m.weight)    # 用该方法初始化权重
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)    # (并)将偏置初始化为0

def make_features(cfg: list):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)

cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], 
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], 
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'], 
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'], 
}

# **后面的变量是一个可变变量，是通过调用vgg函数时所传入的这个字典变量，其中可能包括之前的分类的个数、是否初始化权重的这个布尔变量
def vgg(model_name='vgg16', **kwargs):
    try:
        cfg = cfgs[model_name]
    except:
        print("Warning: model number {} not in cfgs dict!".format(model_name))
        exit(-1)
    model = VGG(make_features(cfg), **kwargs)
    return model
