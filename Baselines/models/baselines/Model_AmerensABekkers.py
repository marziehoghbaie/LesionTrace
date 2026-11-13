"""The code is from https://github.com/ninamalou/TONIC-MICCAI-2024/tree/main"""
import torch
import torch.nn as nn
import torchvision.models as models
##################################################### ADDED ###########################################
class Model_AmerensABekkers(nn.Module):
    def __init__(self, num_classes):
        super(Model_AmerensABekkers, self).__init__()

        # Initialize resnet1 with pretrained weights from torchvision
        self.resnet1 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        resnet1_in_features = self.resnet1.fc.in_features
        self.resnet1.fc = nn.Identity()  # Remove the fully connected layer

        # Initialize resnet2 with pretrained weights from torchvision
        self.resnet2 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        resnet2_in_features = self.resnet2.fc.in_features
        self.resnet2.fc = nn.Identity()  # Remove the fully connected layer

        # New fully connected layer
        # Assuming both resnets have the same number of in_features
        combined_in_features = resnet1_in_features + resnet2_in_features
        self.fc1 = nn.Linear(combined_in_features, num_classes)

    def forward(self, x1, x2):
        out1 = self.resnet1(x1)
        out2 = self.resnet2(x2)
        embed = torch.cat((out1, out2), dim=1)  # Concatenate along the feature dimension
        logits = self.fc1(embed)
        outputs = {'logits': logits, 'embed': embed}
        return outputs
