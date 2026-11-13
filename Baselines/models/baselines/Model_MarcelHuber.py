"""The code is from https://github.com/marceljhuber/mario-miccai2024/tree/main"""

import torch
import torch.nn as nn


class Model_MarcelHuber(nn.Module):
    def __init__(self, num_classes, pretrained):
        super(Model_MarcelHuber, self).__init__()
        self.resnet = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg', pretrained=pretrained)
        self.activation = nn.LeakyReLU()
        self.fc1 = nn.Linear(768 * 2, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, img1, img2):
        x1 = self.resnet.forward_features(img1) # dict_keys(['x_norm_clstoken', 'x_norm_regtokens', 'x_norm_patchtokens', 'x_prenorm', 'masks'])
        cls1 = x1['x_norm_clstoken']
        features_t0 = x1['x_norm_patchtokens']

        x2 = self.resnet.forward_features(img2)

        cls2 = x1['x_norm_clstoken']
        features_t1 = x2['x_norm_patchtokens']

        x = torch.cat((cls1, cls2), dim=1)
        x = self.fc1(x)
        embed = self.activation(x)
        logits = self.fc2(embed)
        outputs = {'logits': logits, 'embed': embed, 'features_t0': features_t0, 'features_t1': features_t1}
        return outputs



if __name__ == '__main__':
    model = Model_MarcelHuber(num_classes=2, pretrained=False)
    img1 = torch.randn(2, 3, 224, 224)
    img2 = torch.randn(2, 3, 224, 224)
    outputs = model(img1, img2)
    print(outputs.keys())