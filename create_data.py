import numpy as np
import csv
import argparse
import os

import cv2

def create_data_frcnn():
    #go through each frame and label it accordingly with the csv
    frame_labels = read_csv('data/data.csv')
    final_labels = []
    #print(frame_labels)
    
    vid = cv2.VideoCapture('data/test.avi')
    i = 0
    start = 0
    while True:
        read, frame = vid.read()
        if not read or i==4500+1:
            break

        name = 'frame'+str(i)+'.jpg'
        cv2.imwrite('data/train/'+name, frame)

        print('reading frame %i...' % i)
        for j in range(start,len(frame_labels)):
            frame_label = frame_labels[j]
            if(frame_label[1] != i):
                start = j
                break
            #do logic
            final_label = []
            final_label.append('../data/train/'+name)
            #write boundaries
            for x in range(8,12):
                final_label.append(str(int(frame_label[x])))
            final_label.append('person')
            print(final_label)
            final_labels.append(final_label)

        i += 1
    with open('data/train.csv', 'w') as csvfile:
        fw = csv.writer(csvfile)
        for label in final_labels:
            fw.writerow(label)

def create_data_vov():
    frame_labels = read_csv('data/data.csv')

    datapath = 'vov_data'
    if not os.path.exists(datapath):
        os.mkdir(datapath)
    if not os.path.exists(datapath+'/train'):
        os.mkdir(datapath+'/train')
    if not os.path.exists(datapath+'/val'):
        os.mkdir(datapath+'/val')

    vid = cv2.VideoCapture('data/test.avi')
    i = 0
    start = 0
    while True:
        read, frame = vid.read()
        if not read or i==4500+1:
            break 
        print('Creating data for frame %i.' % i) 
        for j in range(start,len(frame_labels)):
            frame_label = frame_labels[j]
            if(frame_label[1] != i):
                start = j
                break
            type = 'person'
            tl = (int(frame_label[8]) if frame_label[8] >= 0 else 0, 
                  int(frame_label[9]) if frame_label[9] >= 0 else 0)

            br = (int(frame_label[10]) if frame_label[10] >= 0 else 0, 
                  int(frame_label[11]) if frame_label[11] >= 0 else 0)
            crop = frame[tl[1]:br[1],tl[0]:br[0]]
            name = str(j)+'.jpg'
            print(name)
            savepaths = [datapath+'/train/'+type+'/', datapath+'/val/'+type+'/']
            for savepath in savepaths:
                if not os.path.exists(savepath):
                    os.mkdir(savepath)
            cv2.imwrite(savepaths[0]+name,crop)
            cv2.imwrite(savepaths[1]+name,crop)
        print('Created data for frame %i' % i)
        print('----------')
        i += 1
    print('Done creating data.')
def read_csv(path):
    data = []
    with open(path, 'r') as f:
        r = csv.reader(f)
        for row in r:
            row = list(map(float, row))
            data.append(row)
    return data

parser = argparse.ArgumentParser()
parser.add_argument('network', help='Algorithm of data creation for a certain network')

args = parser.parse_args()

if(args.network == 'frcnn'):
    create_data_frcnn()
elif(args.network == 'vov'):
    create_data_vov()
