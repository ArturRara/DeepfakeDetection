from torch import nn
import torch

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = input + shortcut
        return nn.ReLU()(input)

class ResNet18(nn.Module):
	def __init__(self, in_channels, outputs=1000):
		super().__init__()

		self.layer0 = nn.Sequential(
			nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
			nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU()
		)

		self.layer1 = nn.Sequential(
			Block(64, 64, downsample=False),
			Block(64, 64, downsample=False)
		)

		self.layer2 = nn.Sequential(
			Block(64, 128, downsample=True),
			Block(128, 128, downsample=False)
		)

		self.layer3 = nn.Sequential(
			Block(128, 256, downsample=True),
			Block(256, 256, downsample=False)
		)


		self.layer4 = nn.Sequential(
			Block(256, 512, downsample=True),
			Block(512, 512, downsample=False)
		)

		self.gap = torch.nn.AdaptiveAvgPool2d(1)
		self.fc = torch.nn.Linear(512, outputs)

	def forward(self, input):
		input = self.layer0(input)
		input = self.layer1(input)
		input = self.layer2(input)
		input = self.layer3(input)
		input = self.layer4(input)
		input = self.gap(input)
		input = torch.flatten(input)
		input = self.fc(input)

		return input
#resnet18 = ResNet18(3, outputs=1000)