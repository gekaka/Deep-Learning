下载数据集:
1. LeNet使用的是CIFAR-10数据集, 在LeNet/train.py中运行即可下载(即gitignore中的data/)
2. AlexNet, GoogLeNet, ResNet, ResNext, VGGNet, vision_transformer(ViT)中使用的均为花分类数据集
   (即gitignore中的data_set/)
   下载地址: https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
   如果下载不了的话可以通过百度云链接下载: https://pan.baidu.com/s/1QLCTA4sXnQAw_yvxPj9szg 提取码:58p0
   下载后需使用其中的split_data.py文件对数据集中的图片进行分类处理

训练得到的权重文件可保存在指定文件夹
预训练文件的下载地址在对应的.py文件中, 如有需要可自行下载
若有需要, 可自行改变训练得到的权重文件和预训练文件

用于图像分类的预测图片可自行下载预测, 这里不予提供, 使用时将用于预测的.py文件中的预测图片路径进行修改即可