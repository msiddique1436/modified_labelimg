## Usage python3 --input path_to_image_directory --xml path_to_xml_directory --mode 1 for car and 2 for bikes 






from base64 import b64encode, b64decode
from libs.pascal_voc_io import PascalVocWriter
from libs.pascal_voc_io import XML_EXT
import os.path
import sys
import numpy as np
import logging 
import argparse
import cv2
import sys
import cv2 as cv
from label_segmentation import segNpred
import pandas as pd
import os
import glob
import operator
import natsort


parser = argparse.ArgumentParser()
parser.add_argument("-i","--input",dest="input_images", required = True,help="Input Folder")
parser.add_argument("-o","--xml",dest="xml_Dir", required = True,help="XML files Folder")
parser.add_argument("-m","--mode",dest="mode", required = True,help="Input Folder")
args = parser.parse_args()




ePlate=  segNpred()
















class LabelFile(object):
    # It might be changed as window creates. By default, using XML ext
    # suffix = '.lif'
    suffix = XML_EXT

    def __init__(self, filename=None):
        self.shapes = ()
        self.imagePath = None
        self.imageData = None
        self.verified = False

    def savePascalVocFormat(self, filename, imagePath, imageData,image,apprects,
                            lineColor=None, fillColor=None, databaseSrc=None):
        imgFolderPath = os.path.dirname(imagePath)
        imgFolderName = os.path.split(imgFolderPath)[-1]
        imgFileName = os.path.basename(imagePath)
        #imgFileNameWithoutExt = os.path.splitext(imgFileName)[0]
        # Read from file path because self.imageData might be empty if saving to
        # Pascal format
##        image = QImage()
##        image.load(imagePath)
        imageShape = [image.shape[0], image.shape[1],
                      1 if len(image.shape)==2 else 3]#image.height(), image.width()
        writer = PascalVocWriter(imgFolderName, imgFileName,
                                 imageShape, localImgPath=imagePath)
        writer.verified = self.verified

        for rec in apprects:
##            points = shape['points']
##            label = shape['label']
              label= "char"
            # Add Chris
              difficult = 3#int(shape['difficult'])
##            bndbox = LabelFile.convertPoints2BndBox(points)
              writer.addBndBox(rec[0], rec[1], rec[2], rec[3], label, difficult)

        writer.save(targetFile=filename)
        return

    def toggleVerify(self):
        self.verified = not self.verified

    ''' ttf is disable
    def load(self, filename):
        import json
        with open(filename, 'rb') as f:
                data = json.load(f)
                imagePath = data['imagePath']
                imageData = b64decode(data['imageData'])
                lineColor = data['lineColor']
                fillColor = data['fillColor']
                shapes = ((s['label'], s['points'], s['line_color'], s['fill_color'])\
                        for s in data['shapes'])
                # Only replace data after everything is loaded.
                self.shapes = shapes
                self.imagePath = imagePath
                self.imageData = imageData
                self.lineColor = lineColor
                self.fillColor = fillColor

    def save(self, filename, shapes, imagePath, imageData, lineColor=None, fillColor=None):
        import json
        with open(filename, 'wb') as f:
                json.dump(dict(
                    shapes=shapes,
                    lineColor=lineColor, fillColor=fillColor,
                    imagePath=imagePath,
                    imageData=b64encode(imageData)),
                    f, ensure_ascii=True, indent=2)
    '''

    @staticmethod
    def isLabelFile(filename):
        fileSuffix = os.path.splitext(filename)[1].lower()
        return fileSuffix == LabelFile.suffix

    @staticmethod
    def convertPoints2BndBox(points):
        xmin = float('inf')
        ymin = float('inf')
        xmax = float('-inf')
        ymax = float('-inf')
        for p in points:
            x = p[0]
            y = p[1]
            xmin = min(x, xmin)
            ymin = min(y, ymin)
            xmax = max(x, xmax)
            ymax = max(y, ymax)

        # Martin Kersner, 2015/11/12
        # 0-valued coordinates of BB caused an error while
        # training faster-rcnn object detector.
        if xmin < 1:
            xmin = 1

        if ymin < 1:
            ymin = 1

        return (int(xmin), int(ymin), int(xmax), int(ymax))

lb=LabelFile()
for pat in glob.glob("{}/*.jpg".format(args.input_images)):
##    print("Path {}".format(pat),"mode",args.mode)
    plate=cv.imread(pat)
    fr_count=0
    if int(args.mode)==1:
        boxes_list=ePlate.segNpred_Cars(plate,fr_count)
        lb.savePascalVocFormat("{}/{}.xml".format(args.xml_Dir,pat.split('/')[-1].split('.')[0]),pat,None,plate,boxes_list,'#aa0000',None,None)
    elif int(args.mode)==2:
##        print("Here")
        boxes_list,dst_b=ePlate.segNpred_Bikes(plate,fr_count)
        cv.imwrite(pat,dst_b)
        lb.savePascalVocFormat("{}/{}.xml".format(args.xml_Dir,pat.split('/')[-1].split('.')[0]),pat,None,dst_b,boxes_list,'#aa0000',None,None)
##with open("my_xml/my.xml",'a') as f:##    pass
##lb.savePascalVocFormat("my_xmls/my.xml","home/sohail/1.jpg",None,np.zeros((128,128,3),'uint8'),[[10,10,40,10]],None,None,None)# filename, imagePath, imageData,image,apprects,lineColor=None, fillColor=None, databaseSrc=None

