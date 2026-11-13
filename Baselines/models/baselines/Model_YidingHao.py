import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchvision import models
from torchvision.models import ConvNeXt_Large_Weights, convnext_large, vit_b_16

"""The code is taken from https://github.com/lumine-1/MARIO_Project/tree/main"""

class MultiHeadAttention(nn.Module):
    def __init__(self, in_dim, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        assert in_dim % num_heads == 0, "in_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads

        self.query = nn.Linear(in_dim, in_dim)
        self.key = nn.Linear(in_dim, in_dim)
        self.value = nn.Linear(in_dim, in_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, seq_len, in_dim = x.size()

        query = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attention = self.softmax(torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5))
        attention_output = torch.matmul(attention, value).transpose(1, 2).contiguous().view(batch_size, seq_len, in_dim)

        return attention_output


class Model_YidingHao(nn.Module):
    def __init__(self, num_classes, pretrained):
        super(Model_YidingHao, self).__init__()
        self.cnn = models.convnext_large(pretrained=pretrained)
        self.cnn.classifier[2] = nn.Linear(self.cnn.classifier[2].in_features, 256)
        """
        x = self.features(x) # shape: torch.Size([1, 1536, 7, 7])
        x = self.avgpool(x)
        x = self.classifier(x)
        """
        # 添加多头注意力层
        self.attention = MultiHeadAttention(256)

        # 添加分类器层
        self.classifier = nn.Sequential(
            nn.Linear(256 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward_one(self, x):
        # 将单通道图像复制为三通道图像
        x = x.repeat(1, 3, 1, 1)
        cnn_output = self.cnn(x)
        # 应用注意力机制
        attention_output = self.attention(cnn_output.unsqueeze(1)).squeeze(1)

        return attention_output

    def forward(self, x1, x2):
        output1 = self.forward_one(x1)
        output2 = self.forward_one(x2)

        # 将两个输出拼接在一起
        embed = torch.cat((output1, output2), dim=1)

        # 分类器输出
        logits = self.classifier(embed)
        outputs = {'logits': logits, 'embed': embed}
        return outputs

class Model_YidingHao4Vis(nn.Module):
    def __init__(self, num_classes, pretrained):
        super(Model_YidingHao4Vis, self).__init__()
        self.cnn = models.convnext_large(pretrained=pretrained)
        self.cnn.classifier[2] = nn.Linear(self.cnn.classifier[2].in_features, 256)
        """
        x = self.features(x) # shape: torch.Size([1, 1536, 7, 7])
        x = self.avgpool(x)
        x = self.classifier(x)
        """
        # 添加多头注意力层
        self.attention = MultiHeadAttention(256)

        # 添加分类器层
        self.classifier = nn.Sequential(
            nn.Linear(256 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward_one(self, x):
        # 将单通道图像复制为三通道图像
        x = x.repeat(1, 3, 1, 1)
        features = self.cnn.features(x) # shape: torch.Size([1, 1536, 7, 7])
        cnn_output = self.cnn(x)

        # 应用注意力机制
        attention_output = self.attention(cnn_output.unsqueeze(1)).squeeze(1)

        return attention_output, features

    def forward(self, x1, x2):
        output1, features1 = self.forward_one(x1)
        output2, features2 = self.forward_one(x2)

        # 将两个输出拼接在一起
        embed = torch.cat((output1, output2), dim=1)

        # 分类器输出
        logits = self.classifier(embed)
        outputs = {'logits': logits, 'embed': embed, 'features1': features1, 'features2': features2}
        return outputs


class Model_YidingHaoNoAttn(nn.Module):
    def __init__(self, num_classes, pretrained):
        super(Model_YidingHaoNoAttn, self).__init__()
        self.cnn = models.convnext_large(pretrained=pretrained)
        self.cnn.classifier[2] = nn.Linear(self.cnn.classifier[2].in_features, 256)
        """
        x = self.features(x) # shape: torch.Size([1, 1536, 7, 7])
        x = self.avgpool(x)
        x = self.classifier(x)
        """

        # 添加分类器层
        self.classifier = nn.Sequential(
            nn.Linear(256 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward_one(self, x):
        # 将单通道图像复制为三通道图像
        x = x.repeat(1, 3, 1, 1)
        cnn_output = self.cnn(x)

        return cnn_output

    def forward(self, x1, x2):
        output1 = self.forward_one(x1)
        output2 = self.forward_one(x2)

        # 将两个输出拼接在一起
        embed = torch.cat((output1, output2), dim=1)

        # 分类器输出
        logits = self.classifier(embed)
        outputs = {'logits': logits, 'embed': embed}
        return outputs

if __name__ == '__main__':
    model = Model_YidingHaoNoAttn(3, False)

    inp = torch.randn(1, 1, 224, 224)
    model(inp, inp)