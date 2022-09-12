from torch.utils.data import Dataset
from data_preparation import load_images
from torchvision.transforms import Normalize, RandomAffine
import random
import torch
import torchvision.transforms.functional as TF
import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2

mean=[0.5, 0.5, 0.5]                             
std=[0.5, 0.5, 0.5]
normalize_transform = A.Normalize(mean, std)


Harshil_Albumentations = A.Compose(
    [
        A.ImageCompression(),
        A.GaussNoise(p = 0.1) ,
        A.GaussianBlur(p = 0.05) ,
        A.RandomGamma(p = 0.5) ,
        A.HorizontalFlip(),
        A.OneOf([ 
        A.RandomBrightnessContrast(), 
        A.FancyPCA() , 
        A.HueSaturationValue()], p = 0.7),
        A.ShiftScaleRotate(p = 0.5)
    ]
)

Selimsef_Albumentations = A.Compose(
    [A.ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
    A.GaussNoise(p=0.1),
    A.GaussianBlur(blur_limit=3, p=0.05),
    A.HorizontalFlip(),
    A.OneOf([
        A.IsotropicResize(max_side=256, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
        A.IsotropicResize(max_side=256, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
        A.IsotropicResize(max_side=256, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
    ], p=1),
    A.PadIfNeeded(min_height=256, min_width=256, border_mode=cv2.BORDER_CONSTANT),
    A.OneOf([A.RandomBrightnessContrast(), A.FancyPCA(), A.HueSaturationValue()], p=0.7),
    A.ToGray(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
    ]
)

class ImgDataset(Dataset):
    def __init__(self,df,frames=10,path=r'C:\Users\artur\Downloads\faceforensics_frames',emotion=None,resize=None,transform=None):
        self.train_list = df.reset_index()
        self.frames = frames
        self.path = path
        self.emotion = emotion
        self.resize = resize
        self.transform = transform

    def __getitem__(self, index):
        filename = self.train_list['filename'][index]
        label = self.train_list['category'][index]
        images,labels = load_images(filename, label, self.frames,self.path)
        
        images = self.images_transform(images)
        return images,labels

    def __len__(self):
        return len(self.train_list)

    def resize_image(image, size):
        img = image.resize((size, size))
        return img
    
    def images_transform(self,images):
        if(self.resize!=None):
            for i in range(self.frames):
                images[i] = self.resize_image(images[i],self.resize)

        if(self.transform == "Harshil_Albumentations"):
            for i in range(self.frames):
                images[i] = Harshil_Albumentations(image = images[i])['image']
        elif(self.transform == "Selimsef_Albumentations"):
            for i in range(self.frames):
                images[i] = Harshil_Albumentations(image = images[i])['image']
        
        if(self.emotion == 'Normal'):
            for i in range(self.frames):
                transform = A.Affine(translate_px=[-8,8]) if (random.random() < 0.2) else \
                    A.Affine(translate_px=[-2,2])
                images[i] = transform(image = images[i])['image']
        elif(self.transform == 'Negative'):
            for i in range(self.frames):
                transform = A.Affine(translate_px=[-16,16]) if (random.random() < 0.1) else \
                    A.Affine(translate_px=[-2,2])
                images[i] = transform(images[i])['image']
                images[i] = A.brightness_contrast_adjust(images[i],1.4)['image']
        
        for i in range(self.frames):
            images[i] = normalize_transform(image = images[i])["image"]
            tensor  = ToTensorV2()
            images[i] = tensor(image = images[i])["image"]
            
        return images



if __name__ == '__main__':

    #gaussian_walk = np.ceil(torch.normal(0,1,size=[self.frames,2]))
    print("end")


    

