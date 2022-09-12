from __future__ import print_function, division
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader

import os
import cv2
import argparse
from numpy.random import RandomState
from torch.utils.data import Dataset


def face_foren_videos_to_frames(path=r'C:\Users\artur\Downloads\faceforensics',output_path=r'C:\Users\artur\Downloads\faceforensics_frames'):
    PATH = path
    OUTPUT_PATH = output_path
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
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
        if not frame_capture(video_PATH,images_PATH,name):
            video_df.drop(row.index,axis=0,inplace=True)
    video_df.to_csv(os.path.join(output_path,'metadata.csv'),index=False)


def dfdc_to_frames(path=r'C:\Users\artur\Downloads\dfdc\dfdc_train_part_49\dfdc_train_part_49',output_path=r'C:\Users\artur\Downloads\dfdc_frames'):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(os.path.join(output_path,"FAKE")):
        os.mkdir(os.path.join(output_path,"FAKE"))
        os.mkdir(os.path.join(output_path,"REAL"))
    video_df = pd.read_csv(r'C:\Users\artur\Downloads\dfdc\New_DF.csv')
    
    for index, row in video_df.iterrows():
        filename = row['video']
        category = row['label']
        images_PATH = os.path.join(output_path, category , filename[:-4]+'_frames')
        name = filename[:-4]+'_img_'
        video_PATH = os.path.join(path, filename)
        if not frame_capture(video_PATH,images_PATH,name):
            video_df.drop(row.index,axis=0,inplace=True)
    video_df.to_csv(os.path.join(output_path,'metadata.csv'),index=False)


def frame_capture(path, output_path, name, frames_num=10):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    vidobj = cv2.VideoCapture(path)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    i,j = 0,0
    while i < frames_num:
        success, image = vidobj.read()
        if success:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 6)
        else:
            j+=1 

        if len(faces)<1:
            j+=1
            if(j>1000):
                if(i>0):
                    while i < frames_num:
                        cv2.imwrite("%s/%s_%d.jpg" % (output_path, name, i), cropped_face)
                        i+=1
                else:
                    return False

            continue
        for (x,y,w,h) in faces:
            cropped_face = image[y:y + h, x:x + w]
            cropped_face = cv2.resize(cropped_face, (256, 256))
            cv2.imwrite("%s/%s_%d.jpg" % (output_path, name, i), cropped_face)
        i +=1
    vidobj.release()
    return True


def load_images(filename, label,frames_number=10,path=r'C:\Users\artur\Downloads\faceforensics_frames'):
    images_PATH= path
    frames_path = os.path.join(images_PATH,label,filename[:-4]+'_frames')
    
    i=0
    X = []
    while i <frames_number:
        p = os.path.join(frames_path, filename[:-4]+'_img__'+str(i)+'.jpg')
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        X.append(img)
        i+=1
        
    y = 0 if (label=='original' or label=='REAL') else 1
    y = torch.tensor([y]*frames_number)
    
    return X,  y


if __name__ == '__main__':
    print("start")
    #face_foren_videos_to_frames()
    #dfdc_to_frames()
    print("end")
