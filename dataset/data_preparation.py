from __future__ import print_function, division
import os
import torch
import pandas as pd
from torchvision.transforms import Normalize
import os
import cv2

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize_transform = Normalize(mean, std)

def videos_to_frames():
    PATH=r'C:\Users\MSI\Downloads\faceforensics'
    OUTPUT_PATH=r'C:\Users\MSI\Downloads\faceforensicsimages'
    if not os.path.exists(os.path.join(OUTPUT_PATH,"Face2Face")):
        os.mkdir(os.path.join(OUTPUT_PATH,"Face2Face"))
        os.mkdir(os.path.join(OUTPUT_PATH,"deepfakes"))
        os.mkdir(os.path.join(OUTPUT_PATH,"original"))

    face2face = os.listdir(os.path.join(PATH,'manipulated_sequences', 'Face2Face','c23', 'videos'))
    deepfakes = os.listdir(os.path.join(PATH,'manipulated_sequences', 'deepfakes','c23', 'videos'))
    original = os.listdir(os.path.join(PATH,'original_sequences','youtube','c23', 'videos'))
    label = ['Face2Face']*1000+['deepfakes']*1000+['original']*1000
    video_df = pd.DataFrame()
    video_df['filename'] = face2face+deepfakes+original
    video_df['category'] = label
    video_df.to_csv(r'C:\Users\MSI\Downloads\faceforensicsimages\metadata.csv')

    print(video_df.iterrows())

    for index, row in video_df.iterrows():
        filename=row['filename']
        category=row['category']
        cap = ''
        if category=='Face2Face':
            cap='manipulated_sequences\Face2Face'
        elif category=='deepfakes':
            cap='manipulated_sequences\deepfakes'
        else:
            cap='original_sequences\youtube'
        print(filename)
        video_PATH = os.path.join(PATH, cap, 'c23', 'videos',filename)
        images_PATH = os.path.join(OUTPUT_PATH,category , filename[:-4]+'_frames')
        
        name = filename[:-4]+'_img_'
        print(video_PATH,images_PATH,name)
        frame_capture(video_PATH,images_PATH,name)
    

def frame_capture(path, output_path, name, start_counter=0):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    vidobj = cv2.VideoCapture(path)

    count = 0
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    while count<200:
        success, image = vidobj.read()
        if not success: 
             break
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
       
        faces = face_cascade.detectMultiScale(gray, 1.3, 6)
        
        for (x,y,w,h) in faces:
            cropped_face = image[y:y + h, x:x + w]
            cropped_face = cv2.resize(cropped_face, (256, 256))
            cv2.imwrite("%s/%s_%d.jpg" % (output_path, name, (start_counter + count)), cropped_face)
        count += 1
    vidobj.release()


def load_images(filename, label):
    images_PATH=r'C:\Users\MSI\Downloads\faceforensicsimages'
    frames_path = os.path.join(images_PATH,label,filename[:-4]+'_frames')
    print(frames_path)
    #list = os.listdir(frames_path)
    #frames_count = len(list)
    i,j=0,0
    X = torch.zeros((100, 3, 256, 256))
    while i <100:
        p = os.path.join(frames_path, filename[:-4]+'_img__'+str(j)+'.jpg')
        img = cv2.imread(p)
        if isinstance(img,type(None)):
            j+=1
            continue
        img = torch.tensor(img).float()
        img = normalize_transform(img/255)
        X[i] = img
        i+=1
        j+=1
    y = 0 if label=='original' else 1
    y = torch.tensor([y]*100)
    return X,  y


if __name__ == '__main__':
    videos_to_frames()