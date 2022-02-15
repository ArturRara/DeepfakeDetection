from torch import nn
import torch


class MesoNet(nn.Module):
    
	def __init__(self):
		super(MesoNet, self).__init__()

		self.conv1 = nn.Conv2d(3, 8, 3, padding=1, bias=False)
		self.relu1 = nn.ReLU()
		self.bn1 = nn.BatchNorm2d(8)
		self.maxpooling1 = nn.MaxPool2d(2, 2)

		self.conv2 = nn.Conv2d(8, 8, 5, padding=2, bias=False)
		self.relu2 = nn.ReLU()
		self.bn2 = nn.BatchNorm2d(8)
		self.maxpooling2 = nn.MaxPool2d(kernel_size=(2, 2))

		self.conv3 = nn.Conv2d(8, 16, 5, padding=2, bias=False)
		self.relu3 = nn.ReLU()
		self.bn3 = nn.BatchNorm2d(16)
		self.maxpooling3 = nn.MaxPool2d(kernel_size=(2, 2))

		self.conv4 = nn.Conv2d(16, 16, 5, padding=2, bias=False)
		self.relu4 = nn.ReLU()
		self.bn4 = nn.BatchNorm2d(16)
		self.maxpooling4 = nn.MaxPool2d(kernel_size=(4, 4))

		self.flatten = nn.Flatten()
		self.dropout1 = nn.Dropout2d(0.5)
		self.fc1 = nn.Linear(16*8*8, 16)
		self.leakyrelu = nn.LeakyReLU(negative_slope=0.1)
		self.dropout2 = nn.Dropout2d(0.5)
		self.fc2 = nn.Linear(16, 2)

	def forward(self, input):
		x = self.conv1(input) #(8, 256, 256)
		x = self.relu1(x)
		x = self.bn1(x)
		x = self.maxpooling1(x) 

		x = self.conv2(x) 
		x = self.relu2(x)
		x = self.bn2(x)
		x = self.maxpooling2(x) 

		x = self.conv3(x) 
		x = self.relu3(x)
		x = self.bn3(x)
		x = self.maxpooling3(x) 

		x = self.conv4(x)
		x = self.relu4(x)
		x = self.bn4(x)
		x = self.maxpooling4(x)

		x = self.flatten(x)
		x = self.dropout1(x)
		x = self.fc1(x) 
		x = self.leakyrelu(x)
		x = self.dropout2(x)
		x = self.fc2(x)

		return x


class simple_model(nn.Module):
    def __init__(self, num_class=1):
        super(simple_model, self).__init__()
        
        self.conv = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(32),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
                                  nn.BatchNorm2d(64),
                                  nn.Dropout(0.2))
        
        self.fc = nn.Sequential(nn.Linear(64*10*5, 128),
                                nn.ReLU(inplace=True),
                                nn.BatchNorm1d(128),
                                nn.Dropout(0.2),
                                nn.Linear(128, num_class),
                                nn.Sigmoid())
        
        
    def forward(self, x):
        
        x = self.conv(x)
        x = torch.flatten(x, 1)        
        x = self.fc(x)
        
        return x