from torch import nn
from torch.optim import Adam

class MesoNet(nn.Module):
    
	def __init__(self):
		super(MesoNet, self).__init__()

		self.layer0 = nn.Sequential(
			nn.Conv2d(3, 8, 3, padding=1, bias=False),
			nn.ReLU(),
			nn.BatchNorm2d(8),
			nn.MaxPool2d(2, 2)
		)
		self.layer1 = nn.Sequential(
			nn.Conv2d(8, 8, 5, padding=2, bias=False),
			nn.ReLU(),
			nn.BatchNorm2d(8),
			nn.MaxPool2d(kernel_size=(2, 2))
		)
		
		self.layer2 = nn.Sequential(
			nn.Conv2d(8, 16, 5, padding=2, bias=False),
			nn.ReLU(),
			nn.BatchNorm2d(16),
			nn.MaxPool2d(kernel_size=(2, 2))
		)

		self.layer3 = nn.Sequential(
			nn.Conv2d(16, 16, 5, padding=2, bias=False),
			nn.ReLU(),
			nn.BatchNorm2d(16),
			nn.MaxPool2d(kernel_size=(4, 4))
		)

		self.layer4 = nn.Sequential(
			nn.Flatten(),
			nn.Dropout2d(0.5),
			nn.Linear(16*8*8, 16),
			nn.LeakyReLU(negative_slope=0.1),
			nn.Dropout2d(0.5),
			nn.Linear(16, 2)
		)

	def forward(self, input):
		input = self.layer0(input)
		input = self.layer1(input)
		input = self.layer2(input)
		input = self.layer3(input)
		input = self.layer4(input)

		return input

