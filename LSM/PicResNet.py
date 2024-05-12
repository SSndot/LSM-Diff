from PIL import Image
import numpy as np
from params import args
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet = models.resnet50()
resnet.load_state_dict(torch.load('../model/resnet50.pth'))
resnet = resnet.to(device)


def extract_resnet_features(image_path):
    # 创建图像转换函数
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小为224x224
        transforms.Grayscale(num_output_channels=3),  # 将灰度图像转换为3通道的伪彩色图像
        transforms.ToTensor(),   # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])

    # 加载图像并进行预处理
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # 添加批次维度

    # 将模型设置为评估模式
    resnet.eval()

    # 使用ResNet模型获取特征向量
    with torch.no_grad():
        features = resnet.conv1(image)
        features = resnet.bn1(features)
        features = resnet.relu(features)
        features = resnet.maxpool(features)

        features = resnet.layer1(features)
        features = resnet.layer2(features)
        features = resnet.layer3(features)
        features = resnet.layer4(features)

        features = torch.mean(features, dim=(2, 3))[:, :args.input_size]

    return features


target_set = "video_0/"
image_features = []

for i in range(args.pic_num):
    pic_name = 'frame_' + str(i) + '.jpg'
    img_path = args.file_path + target_set + pic_name
    image_feature = extract_resnet_features(img_path)
    image_features.append(image_feature)

train_set = image_features[:args.train_num]
test_set = image_features[args.train_num:]
