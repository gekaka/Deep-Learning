
import torch
from model import resnet34
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json

data_transform = transforms.Compose(
    [transforms.Resize(224), 
     transforms.CenterCrop(224), 
     transforms.ToTensor(), 
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# load image
img = Image.open("./roses.webp")
# plt.imshow(img)
# [N, C, H, W]
img = data_transform(img)
# expend batch dimension
img = torch.unsqueeze(img, dim=0)

# read class_indict
try:
    json_file = open('./class_indices.json', 'r')
    class_dict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

# create model
model = resnet34(num_classes=5)
# load model weights
model_weight_path = "./resNet34.pth"
missing_keys, unexpected_keys = model.load_state_dict(torch.load(model_weight_path), strict=False)
model.eval()
with torch.no_grad():
    # predict class
    output = torch.squeeze(model(img))
    predict = torch.softmax(output, dim=0)
    predict_cla = torch.argmax(predict).numpy()
print(class_dict[str(predict_cla)], predict[predict_cla].item())
# plt.show()
