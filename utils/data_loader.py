
from torch.utils.data import Dataset
from dataset.data_preparation import load_images

class ImgDataset(Dataset):
    def __init__(self,df):
        self.df = df
        self.filename = df['filename']
        self.labels = df['category']
    def __getitem__(self, index):
        x = self.filename[index]
        y = self.labels[index]
        return load_images(x, y)
    def __len__(self):
        return len(self.df)
    