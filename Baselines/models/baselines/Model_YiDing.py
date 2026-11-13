"""The code is from https://github.com/YIDING4869/MARIO2024/tree/main"""
import torch
import torchvision
from torch import nn
from torchvision import models
import timm


# class Model_YiDing(nn.Module):
#     def __init__(self, num_classes=4, pretrained=True):
#         super(Model_YiDing, self).__init__()
#         # 使用 timm 导入预训练的 ResNet50 模型
#         self.model = timm.create_model('resnet50', pretrained=pretrained)
#
#         # 调整第一个卷积层以接受单通道图像
#         self.model.conv1 = nn.Conv2d(1, self.model.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
#
#         # 移除最后的分类器层
#         self.model.fc = nn.Identity()
#
#         # 获取特征输出大小（这里ResNet50的最后一个卷积层的输出通道数通常是2048）
#         num_features = self.model.num_features
#
#         # 定义用于处理每个特征的全连接层
#         self.fc_single = nn.Sequential(
#             nn.Linear(num_features, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256)
#         )
#
#         # 定义用于处理差异特征的全连接层
#         self.fc_diff = nn.Sequential(
#             nn.Linear(num_features, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256)
#         )
#
#         # 定义最终分类的全连接层
#         self.fc_final = nn.Sequential(
#             nn.Linear(256 * 3, 512),
#             nn.ReLU(),
#             nn.Linear(512, num_classes)
#         )
#
#     def forward_one(self, x):
#         return self.model(x)
#
#     def forward(self, input1, input2):
#         output1 = self.forward_one(input1)
#         output2 = self.forward_one(input2)
#
#         diff = torch.abs(output1 - output2)  # 计算差异
#
#         # 处理每个特征和差异特征
#         feature1 = self.fc_single(output1)
#         feature2 = self.fc_single(output2)
#         feature_diff = self.fc_diff(diff)
#
#         # 连接特征并通过最终分类层
#         embed = torch.cat((feature1, feature2, feature_diff), dim=1)
#         logits = self.fc_final(embed)
#
#         outputs = {'logits': logits, 'embed': embed}
#         return outputs
class Model_YiDing(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super(Model_YiDing, self).__init__()
        # 使用 timm 导入预训练的 ConvNeXt V2-L 模型
        self.model = models.convnext_large(pretrained=pretrained)
        # 调整第一个卷积层以接受单通道图像
        self.model.features[0][0] = nn.Conv2d(
            in_channels=1,  # grayscale input
            out_channels=self.model.features[0][0].out_channels,
            kernel_size=4,
            stride=4
        )
        self.model.classifier = nn.Identity()  # 移除最后一层分类器


        # 定义用于处理每个特征的全连接层
        self.fc_single = nn.Sequential(
            nn.Linear(1536, 512),  # 输入大小取决于模型输出大小
            nn.ReLU(),
            nn.Linear(512, 256)
        )

        # 定义用于处理差异特征的全连接层
        self.fc_diff = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

        # 定义最终分类的全连接层
        self.fc_final = nn.Sequential(
            nn.Linear(256 * 3, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),  # 输出类别数
            # nn.Softmax(dim=1)  # 使用Softmax进行多类别分类
        )

    def forward_one(self, x):
        return self.model(x)

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        output1 = output1.view(output1.size(0), -1)
        output2 = output2.view(output2.size(0), -1)

        diff = torch.abs(output1 - output2)  # 计算差异
        # 处理每个特征和差异特征
        feature1 = self.fc_single(output1)
        feature2 = self.fc_single(output2)
        feature_diff = self.fc_diff(diff)

        # 连接特征并通过最终分类层
        combined_features = torch.cat((feature1, feature2, feature_diff), dim=1)
        out = self.fc_final(combined_features)

        return {'logits':out, 'embed':combined_features}


if __name__ == '__main__':
    model = Model_YiDing(num_classes=4, pretrained=True)
    inp = torch.randn(1, 1, 224, 224)
    print(model(inp, inp))
