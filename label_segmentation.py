#sohail and lakshman new code segmentation + Below changes
#Added limits to reduce the errors using  HEIGHT ,WIDTH, AREA
#Tested with 3 different data sets of 130,170,300
#For Max width focused on characters W, M, Min width focused on character i, 1

#min_height=10px   max_height=43px >>  name in code rec[3]
#min_width= 6px    max_width=28px  >>  name in code rec[2]  
#min_Area=210px    max_Area=1050px >> name in code ar
#Enabling debug flags to display segmentation blocks

import logging 
import argparse
import cv2
import sys
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import pandas as pd
import time
import natsort
import os
import cv2 as cv
import pdb
from sklearn import preprocessing
stp=0
count=0
count1=0
framecount=0
from datetime import datetime
from keras.models import load_model
import os
import glob
import operator
import time
import imutils
from imutils.object_detection import non_max_suppression

class segNpred:

    ###########Init SVM#################33
    
    integers='0123456789'
    #######################################
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

# load the pre-trained EAST text detector
    print("[INFO] loading EAST text detector...")
    net = cv2.dnn.readNet("toloadmodel/frozen_east_text_detection.pb")

    ##########segment and predict#######3
    
    def __init__(self):
        
        print("Initalized segmentation & char recognition")
    

    def segNpred_Cars(self,plate,frameCount):
        
        boxes_list=[]
        img = plate
##        dirs_path=  os.getcwd()
        seg_count=0
        min_area_char=210
        max_area_char=1050
        min_height_char=10
        max_height_char=60
        min_width_char=6
        max_width_char=27
        jointCH_max_heght=100 #FOR PROCESS HEIGHT JOINT CHARECTORS
        jointCH_max_width=100 #FOR WIDTH JOINT CHARECTORS
        ratio=0
        
        print(img.shape)
        img1=cv2.resize(img,(1*img.shape[1],1*img.shape[0]))
    
        #imgshow=cv2.resize(img,(2*img.shape[1],2*img.shape[0]))
        #imgshow0=cv2.resize(img,(2*img.shape[1],2*img.shape[0]))
        area={}
        ycord={}
        datadict={}
        xdict={}
        xlist=[]
        height=[]
        bb=[]
        num=[]
        num_svm=[]
        ocr_acc=[]
        check_errors=0
        grey=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        val=cv2.split(cv2.cvtColor(img1,cv2.COLOR_BGR2HSV))[2]
        ret,img_threshold=cv2.threshold(val, 12, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)#grey

#        st=time.time()
        kernel=np.ones((3,1),np.uint8)
        img_thresholdn=np.zeros((val.shape[0],val.shape[1]),np.uint8)
        np.copyto(img_thresholdn,img_threshold)

        def ero_dil(img_threshold):           #DILATION and EROSION for high width  joint characters
            print("erode started")##########DEBUG
            img_threshold = cv2.dilate(img_threshold,kernel,iterations = 2) #dilate
            img_threshold = cv2.erode(img_threshold,kernel,iterations = 2) #erode
            #cv2.imshow("ERODE THRESLD",img_threshold)#########DEBUG
            return img_threshold 

        im2, contours, hierarchy = cv2.findContours(img_threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        for i,cnt in enumerate(contours):
            approx=cv2.approxPolyDP( cnt, 3, True )
            appRect= cv2.boundingRect(approx) # x,y,w,h
            area[i]=appRect[3]*appRect[3]
            ycord[i]=appRect[1]
            bb.append(appRect)
       #         cv2.rectangle(imgshow,(appRect[0],appRect[1]),(appRect[0]+appRect[2],appRect[1]+appRect[3]),(0,255,0),2)
            #cv2.drawContours(img, contours, -1, (0,255,0), 1)
           # cv2.imshow("rm",imgshow)
            #cv2.waitKey(0)
        maxarea=max(list(area.values()))
        minarea=min(list(area.values()))

        
        #cv2.imshow("np.zeros thrsld",img_thresholdn)
        #cv2.imshow("THRESLD",img_threshold)
        while(1):                       
            #cv2.imshow("Processed image",img_threshold)    # it show the corrent state of image #########DEBUG
            height=[]
            bb=[]
            #print("segmentation process started") ############DEBUG            
            break_heights=0 #  break_height is used to break while loop of geting heights   
            check_errors =check_errors+1   # check_errors to exit the loop of  calling (ero_dil())  erosion and dilatio function (ero_dil)
            im2, contours, hierarchy = cv2.findContours(img_threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            for i,cnt in enumerate(contours):
                approx=cv2.approxPolyDP( cnt, 3, True )
                appRect= cv2.boundingRect(approx) # x,y,w,h
                area[i]=appRect[3]*appRect[3]
                ycord[i]=appRect[1]
                bb.append(appRect)
                #cv2.rectangle(imgshow0,(appRect[0],appRect[1]),(appRect[0]+appRect[2],appRect[1]+appRect[3]),(0,255,0),2)
            #cv2.drawContours(img, contours, -1, (0,255,0), 1)
            #cv2.imshow("rm0",imgshow0)
            #cv2.waitKey(0)
            #maxarea=max(list(area.values()))
            #minarea=min(list(area.values()))
            #newarea={ind:are for ind,are in area.items() if are!=maxarea and are!=minarea and are>10}
                #meanarea=sum(list(newarea.values()))/len(list(newarea.values()))
        
            max_char=1 # max10 is usefull to show  count number of characters detected 
            for j,rec in enumerate(bb):
                ar=rec[2]*rec[3]
         #getting heights in the for loop 
                if ar<maxarea and ar >minarea  and rec[3]>min_height_char and rec[3]<jointCH_max_heght and rec[2]>min_width_char and rec[2]<jointCH_max_width:#and 0.2<ar/float(meanarea)<1.5  # her checking (max and min area) ,H,W
                    #cv2.rectangle(imgshow,(rec[0],rec[1]),(rec[0]+rec[2],rec[1]+rec[3]),(0,255,0),2) ######DEBUG
                    #print(max_char," ","area :",ar," ","height :",rec[3]," ","weidth",rec[2],"rec[0]",rec[0])
                    max_char=max_char+1                       # max10 is usefull to show  count number of characters detected       
                    height.append(rec[3])               # her append the height where we required  
                    datadict[rec[3]]=[rec[0],rec[1],rec[2]]
                    if(check_errors<3):
                        if(rec[2]>=60 or rec[3]>80 ):               #checking Width and Height  for process  dilation and erosion
                            img_threshold =ero_dil(img_threshold)
                            #print("rec[2]=",rec[2])########DEBUG
                            break_heights=1
                            break
            
                elif ar==maxarea or ar==minarea:
                    del area[j]
            #cv2.imshow("rm",imgshow)
            #cv2.waitKey(0)
            
            if(break_heights==0):   # flag1 for exit for  While LOOP
                break

                    
        srtheight=sorted(height)         # sorting height heights
        #print("hts",len(srtheight),"::",srtheight)
        #pdb.set_trace()  
        dx=[abs(x-srtheight[i+1]) for i,x in enumerate(srtheight) if i!=(len(srtheight)-1)]   # getting diffences between heights
        #print("difference:",dx) ######### DEBUG
        nboxes=0
        indx=[]
        acceptableheights=[]
        for ct,j in enumerate(dx):
            if j<=4:           # max  3 diffence is allow
                nboxes+=1
            if j>4 or ct==(len(dx)-1): # out of range add or clear the boxes or both
                #pdb.set_trace()
                if (len(dx)>=8 and nboxes>=4) or (len(dx)<=8):           # L # filtering the characters  using heights  differences
                    indx.append([ct-(nboxes-1),nboxes])
                    #print("added")
                #print("clear")
                nboxes=0
        for l in range(len(indx)):
            for k in range(indx[l][1]+1):             #getting acceptable heights
                #print("Huna-1")
                acceptableheights.append(srtheight[indx[l][0]+k])
        #print(acceptableheights) #########DEBUG
        for rec in bb:           # double checking the characters process and getting the requied  height characters
            ar=rec[2]*rec[3]
            if rec[3] in acceptableheights and 1.4*rec[3] >rec[2] :#and rec[2]>min_width_char and ar <=max_area_char and ar >min_area_char:#filter charecters H,W,A
                xlist.append(rec[0])
                xdict[rec[0]]=[rec[0],rec[1],rec[2],rec[3]]
                #print("Hunnna")
        #pdb.set_trace()      
        xlistsrt=sorted(xlist)
        #print("xlist_size",len(xlistsrt))
        for v,ky in enumerate(xlistsrt):
            #if v>0:print("Cords",(xdict[xlistsrt[v-1]][2]+xdict[xlistsrt[v-1]][0]),"cur",(xdict[xlistsrt[v]][2]+xdict[xlistsrt[v]][0]))
            if v>0 and (xdict[xlistsrt[v-1]][2]+xdict[xlistsrt[v-1]][0])>(xdict[xlistsrt[v]][2]+xdict[xlistsrt[v]][0]):continue
            boxes_list.append([int(xdict[ky][0]),int(xdict[ky][1]),int((xdict[ky][0]+xdict[ky][2])),int((xdict[ky][1]+xdict[ky][3]))])
            
            count=count+1
            seg_count=seg_count+1
    
        return boxes_list
    ########      END of CAR SEGMENTATION CODE     ############
    def image_feed(self,image,itr,otsu):
            #image = cv2.imread(args["image"])
            orig = image.copy()
            orig1=image.copy()
            (H, W) = image.shape[:2]
            y_cor=0
            y_cor_end=0
            # set the new width and height and then determine the ratio in change
            # for both the width and height
            
            (newW, newH) = (96, 96)
            rW = W / float(newW)
            rH = H / float(newH)

            
            # resize the image and grab the new image dimensions
            image = cv2.resize(image, (newW, newH))
            (H, W) = image.shape[:2]
            #cv2.imshow("inside",image)
            #cv2.waitKey(0)
            # construct a blob from the image and then perform a forward pass of
            # the model to obtain the two output layer sets
            blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                    (123.68, 116.78, 103.94), swapRB=True, crop=False)
            start = time.time()
            self.net.setInput(blob)
            (scores, geometry) = self.net.forward(self.layerNames)
            end = time.time()

            # show timing information on text prediction
            print("[INFO] text detection took {:.6f} seconds".format(end - start))

            # grab the number of rows and columns from the scores volume, then
            # initialize our set of bounding box rectangles and corresponding
            # confidence scores
            (numRows, numCols) = scores.shape[2:4]
            rects = []
            boxrects=[]
            confidences = []
            angles_list=[]
            thres=[]

            # loop over the number of rows
            for y in range(0, numRows):
                    # extract the scores (probabilities), followed by the geometrical
                    # data used to derive potential bounding box coordinates that
                    # surround text
                    scoresData = scores[0, 0, y]
                    xData0 = geometry[0, 0, y]
                    xData1 = geometry[0, 1, y]
                    xData2 = geometry[0, 2, y]
                    xData3 = geometry[0, 3, y]
                    anglesData = geometry[0, 4, y]

                    # loop over the number of columns
                    for x in range(0, numCols):
                            # if our score does not have sufficient probability, ignore it
                            if scoresData[x] < 0.5:
                                    continue

                            # compute the offset factor as our resulting feature maps will
                            # be 4x smaller than the input image
                            (offsetX, offsetY) = (x * 4.0, y * 4.0)

                            # extract the rotation angle for the prediction and then
                            # compute the sin and cosine
                            angle = anglesData[x]
                            angles_list.append(angle)
                            cos = 1#np.cos(angle*180/np.pi)
                            sin = 0#np.sin(angle*180/np.pi)
                            #print("Angle is :: ",angle)
                            # use the geometry volume to derive the width and height of
                            # the bounding box
                            h = xData0[x] + xData2[x]
                            w = xData1[x] + xData3[x]

                            # compute both the starting and ending (x, y)-coordinates for
                            # the text prediction bounding box
                            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                            startX = int(endX - w)
                            startY = int(endY - h)

                            # add the bounding box coordinates and probability score to
                            # our respective lists
                            rects.append((startX, startY, endX, endY))
                            boxrects.append(np.int0(cv2.boxPoints((((startX+endX)*rW/2,(startY+endY)*rH/2),(w*rW,h*rH),360-(angle*180)/np.pi))))
                            confidences.append(scoresData[x])

            # apply non-maxima suppression to suppress weak, overlapping bounding
            # boxes
            boxes = non_max_suppression(np.array(rects), probs=confidences)
            box1=non_max_suppression(np.array(boxrects), probs=confidences)
            for (startX, startY, endX, endY) in boxes:
                    # scale the bounding box coordinates based on the respective
                    # ratios
                    startX = int(startX * rW)
                    startY = int(startY * rH)
                    endX = int(endX * rW)
                    endY = int(endY * rH)
                    if startY>y_cor:y_cor=startY
                    if endY>y_cor_end:y_cor_end=endY
                    #print([startX,startY,abs(endX-startX),abs(endY-startY)])
                    #img2= cv2.boundingRect(np.array([startX,startY,abs(endX-startX),abs(endY-startY)]))
                    #img2=np.array([startX,startY,abs(endX-startX),abs(endY-startY)])
                    #print("Rect",img2)
                    #rect = cv2.minAreaRect(img2)
                    
                    #box = cv2.boxPoints(img2)
                    #box = np.int0(box)
                    #print("Box",box)
                    #cv2.drawContours(orig1,[startX,startY,abs(endX-startX),abs(endY-startY)],0,(0,0,255),2)
                    # draw the bounding box on the image
                    ret,_=cv2.threshold(cv2.cvtColor(orig[startY:endY,startX:endX],cv2.COLOR_BGR2GRAY),0,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY)
                    thres.append(ret)
                    cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
            if len(angles_list)==0:return 0,orig,orig1,[int(otsu)],orig.shape[0],None,0
            else:return max(angles_list),orig,orig1,thres,y_cor,y_cor_end,1
    def get_thresholded(self,gry,sto,ycor,ycor_end,ky):
            if ky==1:
                    if len(sto)==2:
                            gry[:ycor,:]=cv2.threshold(gry[:ycor,:],sto[0],255,cv2.THRESH_BINARY)[1]
                            gry[ycor:,:]=cv2.threshold(gry[ycor:,:],sto[1],255,cv2.THRESH_BINARY)[1]
                    elif len(sto)==1:
                            gry=cv2.threshold(gry,sto[0],255,cv2.THRESH_BINARY)[1]
                    elif len(sto)>2:
                            gry[:ycor,:]=cv2.threshold(gry[:ycor,:],sto[0],255,cv2.THRESH_BINARY)[1]
                            gry[ycor:,:]=cv2.threshold(gry[ycor:,:],sto[-1],255,cv2.THRESH_BINARY)[1]
                    else : gry=cv2.threshold(gry,0,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY)[1]
            elif ky==0:gry=cv2.threshold(gry,0,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY)[1]
            return gry
    def get_masks(self,image):
            
            img0=image.copy()
            #img0_next=image.copy()
            img0[0:2,:]=255*np.ones((2,img0.shape[1],3),'uint8')
            img0[-2:,:]=255*np.ones((2,img0.shape[1],3),'uint8')
            img0[:,:2]=255*np.ones((img0.shape[0],2,3),'uint8')
            img0[:,-2:]=255*np.ones((img0.shape[0],2,3),'uint8')
            img0_next=img0.copy()
            #img0_next[0:2,:]=255*np.ones((2,img0.shape[1],3),'uint8')
            #img0_next[-2:,:]=255*np.ones((2,img0.shape[1],3),'uint8')
            tr=[]
            otsu=cv2.threshold(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),0,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY)[0]
##            print("Pre Thresh--{}".format(otsu))

            for i in range(2):
                ang,orig,orig1,sto,ycor,ycor_end,ky=self.image_feed(image,i,otsu)
                tr.append(min(sto))
           
                if i==0:
                    dst_b=imutils.rotate_bound(image,ang*180/np.pi)
                    img0_next=imutils.rotate_bound(img0_next,ang*180/np.pi)
                    gry_2=cv2.cvtColor(img0_next,cv2.COLOR_BGR2GRAY)
                    #gry_2=cv2.blur(gry_2,(5,5))
                elif i==1:
                    otsu_img=cv2.threshold(gry_2,0,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY)[1]
                    simp40=cv2.threshold(gry_2,40,255,cv2.THRESH_BINARY)[1]
                    simp60=cv2.threshold(gry_2,60,255,cv2.THRESH_BINARY)[1]
                    simp90=cv2.threshold(gry_2,90,255,cv2.THRESH_BINARY)[1]
                    simp110=cv2.threshold(gry_2,110,255,cv2.THRESH_BINARY)[1]
                    simp160=cv2.threshold(gry_2,160,255,cv2.THRESH_BINARY)[1]
                    gry_2=self.get_thresholded(gry_2,sto,ycor,ycor_end,ky)
                    
                    thresh_aft=cv2.threshold(cv2.blur(cv2.cvtColor(img0_next,cv2.COLOR_BGR2GRAY),(5,5)),min(tr),255,cv2.THRESH_BINARY)[1]
                    thresh_aft_er=cv2.dilate(cv2.erode(cv2.threshold(cv2.cvtColor(img0_next,cv2.COLOR_BGR2GRAY),min(tr),255,cv2.THRESH_BINARY)[1],np.ones((3,3),np.uint8),iterations=1),np.ones((3,1),np.uint8),iterations=1)
            return dst_b,gry_2,thresh_aft,thresh_aft_er,otsu_img,simp40,simp60,simp90,simp110,simp160         


    def segNpred_Bikes(self,plate,frameCount):
        
        seg_count=0
        box_list=[]
        img0=cv2.resize(plate,(1*plate.shape[1],1*plate.shape[0]))
        #img1=cv2.resize(plate,(2*plate.shape[1],2*plate.shape[0]))
        dst_b,gry_2,thresh_aft,thresh_aft_er,otsu_img,simp40,simp60,simp90,simp110,simp160  =self.get_masks(img0)
        img1=dst_b.copy()
        area={}
        ycord={}
        datadict={}
        clusters=[]
        cluster2=[]
        
        ydict={}
        #xlist=[]
        ylist=[]
        height=[]
        width=[]
        bb=[]
        num=[]
        num_svm=[]
        ocr_acc=[]
        #check_errors=0
        countour_list=[]
        grey=cv2.cvtColor(dst_b,cv2.COLOR_BGR2GRAY)
        #cv2.imshow("deskw",deskew(grey))
        val=cv2.split(cv2.cvtColor(img1,cv2.COLOR_BGR2HSV))[2]
        #cv2.imshow("grey",grey)
        #cv2.imshow("val",val)
        ret,img_threshold=cv2.threshold(grey, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
        #cv2.imshow("bth",img_threshold)
        kernel=np.ones((3,3),np.uint8)
        #img_threshold=cv2.dilate(img_threshold,kernel,iterations=1)
        #img_threshold=cv2.erode(img_threshold,kernel,iterations=1)
    #    cv2.imshow("ath",img_threshold)
    #    print("No of comp",np.amax(cv2.connectedComponents(cv2.bitwise_not(img_threshold))[1]))
        for img_tres in [gry_2,otsu_img,simp40,simp60,simp90,simp110,simp160  ]:#[thresh_aft] thresh_aft_er,otsu_img
            im2, contours, hierarchy = cv2.findContours(img_tres,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)#cv2.RETR_TREE
            countour_list.append(contours)
        for contour in countour_list:
            for i,cnt in enumerate(contour):
                approx=cv2.approxPolyDP( cnt, 3, True )
                appRect= cv2.boundingRect(approx) # x,y,w,h
                
                if appRect not in bb:
                        bb.append(appRect)
                        area[i]=appRect[2]*appRect[3]
                        ycord[i]=appRect[1]

        #cv2.rectangle(img0,(appRect[0],appRect[1]),(appRect[0]+appRect[2],appRect[1]+appRect[3]),(0,255,0),2)
        #cv2.imshow("prev",img0)
    #cv2.drawContours(img, contours, -1, (0,255,0), 1)
        if len(list(area.values()))<3:return None,None,None
        maxarea=max(list(area.values()))
        minarea=min(list(area.values()))
        newarea={ind:are for ind,are in area.items() if are!=maxarea and are!=minarea and are>10}
        #meanarea=sum(list(newarea.values()))/len(list(newarea.values())) 
        newbb=[]
        for j,rec in enumerate(bb):
            ar=rec[2]*rec[3]
            if ar<3000 and ar >minarea and rec[3]>15 and rec[2]>5 and rec[2]<50 and rec[2]<2*rec[3] and rec[3]<50 :#and 0.2<ar/float(meanarea)<1.5 ; ar<maxarea and ar >minarea and rec[3]>5
                cv2.rectangle(img0,(rec[0],rec[1]),(rec[0]+rec[2],rec[1]+rec[3]),(0,255,0),2)
                newbb.append(rec)
                height.append(rec[3])
                width.append(rec[2])
                ylist.append(rec[1]) 
                ydict[rec[1]]=[rec[0],rec[1],rec[2],rec[3]]
                datadict[rec[3]]=[rec[0],rec[1],rec[2]] #currently unused
                #print(rec[1],"::",0.4*img1.shape[0])
                #if rec[1]<0.4*img1.shape[0]:cluster1.append(rec)#[rec[3]]=[rec[0],rec[1],rec[2]]
                #else:cluster2.append(rec)#cluster2[rec[3]]=[rec[0],rec[1],rec[2]] 
            #elif ar==maxarea or ar==minarea:
             #   del area[j]
        srtheight=sorted(height)
        srtwidth=sorted(width)
        dxwd=[abs(x-srtwidth[i+1]) for i,x in enumerate(srtwidth) if i!=(len(srtwidth)-1)]
        dx=[abs(x-srtheight[i+1]) for i,x in enumerate(srtheight) if i!=(len(srtheight)-1)]
        nboxes=0
        nboxeswd=0
        indx=[]
        indxwd=[]
        acceptableheights=[]
        acceptablewidths=[]
        #print("Hts",srtheight)
        #print("dx ::",dx)
        for ct,j in enumerate(dx):
            if j<7:
                nboxes+=1
            if j>=7 or ct==(len(dx)-1):
                if (len(dx)>3 and nboxes>=3) :
                    if j<7:indx.append([ct-(nboxes-1),nboxes])#or (len(dx)<5)
                    elif j>=7:indx.append([ct-(nboxes),nboxes])
                nboxes=0
        for l in range(len(indx)):
            for k in range(indx[l][1]+1):
            #print("Huna-1")
                acceptableheights.append(srtheight[indx[l][0]+k])
        for ct,j in enumerate(dxwd):
            if j<3:
                nboxeswd+=1
            if j>=3 or ct==(len(dxwd)-1):
                if (len(dxwd)>3 and nboxeswd>=3) :indxwd.append([ct-(nboxeswd-1),nboxeswd])#or (len(dx)<5)
                nboxeswd=0
        for l in range(len(indxwd)):
            for k in range(indxwd[l][1]+1):
            #print("Huna-1")
                acceptablewidths.append(srtwidth[indxwd[l][0]+k])

        srtdlist_w_index=sorted(enumerate(ylist),key=operator.itemgetter(1))#this returns a list of tuple (index,value)
        ylistsrt=sorted(ylist)
        #print(srtdlist_w_index)
        ycorind0=0
        itr=0
        for i,tup in enumerate(srtdlist_w_index):#tup=(index,ycor)
            if i==0:ycorind0=tup[0]
            cl=[]
            #continue
            if i>0 and  newbb[tup[0]][1]+(newbb[tup[0]][3])/4>newbb[ycorind0][1]+newbb[ycorind0][3] :#+(newbb[tup[0]][3])/2
                
                for j in range(itr,i):cl.append(newbb[srtdlist_w_index[j][0]])
                itr=i
                clusters.append(cl)
            ycorind0=tup[0]
            if i==len(srtdlist_w_index)-1 and itr!=len(srtdlist_w_index)-1: 
                for k in range(itr,i+1):cl.append(newbb[srtdlist_w_index[k][0]])
                clusters.append(cl)
    #    cv2.imshow("prev",img0)
        #print("Accep",acceptableheights)
        for cct,cluster in enumerate(clusters):
    #        print("clust count",cct)
    #        print(cct,"::",cluster)
            #height=[]
            xlist=[]
            xdict={}
            xend=[]
            for rec in cluster:
                if rec[3] in acceptableheights  :#1.4*rec[3]>rec[2]and rec[3]>=1*rec[2] rec[2]>3: and rec[2] in acceptablewidths
                    if rec[0] not in xlist and rec[0]+1 not in xlist and rec[0]+2 not in xlist and rec[0]-1 not in xlist and rec[0]-2 not in xlist  and rec[0]+3 not in xlist and rec[0]-3 not in xlist and rec[0]+4 not in xlist and rec[0]-4 not in xlist and rec[0]+rec[2] not in xend and rec[0]+rec[2]+1 not in xend and rec[0]+rec[2]+2 not in xend and rec[0]+rec[2]-1 not in xend and rec[0]+rec[2]-2 not in xend  and rec[0]+rec[2]+3 not in xend and rec[0]+rec[2]-3 not in xend:
                        xlist.append(rec[0])
                        xend.append(rec[0]+rec[2])
                        xdict[rec[0]]=[rec[0],rec[1],rec[2],rec[3]]
                    else:
                        xdict[xlist[np.argmin(abs(np.array(xlist)-rec[0]))]][3]=max(rec[3],xdict[xlist[np.argmin(abs(np.array(xlist)-rec[0]))]][3])
                        xdict[xlist[np.argmin(abs(np.array(xlist)-rec[0]))]][0]=min(xlist[np.argmin(abs(np.array(xlist)-rec[0]))],rec[0])
                #ylist.append(rec[1])
                #ydict[rec[1]]=[rec[0],rec[1],rec[2],rec[3]]
                    
            #print("Huna")
            xlistsrt=sorted(xlist)
            ylistsrt=sorted(ylist)
            #print(sorted(ylist))
            for v,ky in enumerate(xlistsrt):
                #if v>0:print("Cords",(xdict[xlistsrt[v-1]][2]+xdict[xlistsrt[v-1]][0]),"cur",(xdict[xlistsrt[v]][2]+xdict[xlistsrt[v]][0]))
                if v>0 and (xdict[xlistsrt[v-1]][2]+xdict[xlistsrt[v-1]][0])>(xdict[xlistsrt[v]][2]+xdict[xlistsrt[v]][0]):continue
                box_list.append([int(xdict[ky][0]),int(xdict[ky][1]),int((xdict[ky][0]+xdict[ky][2])),int((xdict[ky][1]+xdict[ky][3]))])
                crop=grey[int(xdict[ky][1])-min(0,int(xdict[ky][1])):int((xdict[ky][1]+xdict[ky][3]))+min(0,(grey.shape[0]-xdict[ky][1]-xdict[ky][3])),int(xdict[ky][0])-min(0,int(xdict[ky][0])):int((xdict[ky][0]+xdict[ky][2]))+min(0,(grey.shape[1]-xdict[ky][0]-xdict[ky][2]))]#img_threshold
                seg_count=seg_count+1
        return box_list,dst_b
