# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np
import os, math
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import scipy
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import skimage, skimage.morphology
import cv2
from PIL import Image, ImageDraw
from scipy.ndimage.interpolation import rotate
from skimage import color, measure

import scipy.ndimage

from numba import jit, autojit
from utils.generic_utils import get_from_module
import random
def get(identifier):
    return get_from_module(identifier, globals(), 'local_utils')

class myobj(object):
    pass

def imread(imgfile):
    return np.asarray(Image.open(imgfile))

def writeImg(array, savepath):
    im = Image.fromarray(array.astype(np.uint8))
    im.save(savepath)


def SaveFigureAsImage(fileName,fig=None,**kwargs):
    ''' Save a Matplotlib figure as an image without borders or frames.
       Args:
            fileName (str): String that ends in .png etc.

            fig (Matplotlib figure instance): figure you want to save as the image
        Keyword Args:
            orig_size (tuple): width, height of the original image used to maintain
            aspect ratio.
			SaveFigureAsImage('out.png', plt.gcf() )

		# Here plt.gcf() gets a handle to your current figure. You could also save this like so:
		# myFig = figure(2)  if you needed the second figure.

		# To Plot with a different aspect ratio, use orig_size
		SaveFigureAsImage('out2.png',plt.gcf(), orig_size=(640,480) )

    '''
    fig_size = fig.get_size_inches()
    w,h = fig_size[0], fig_size[1]
    fig.patch.set_alpha(0)
    if kwargs.has_key('orig_size'): # Aspect ratio scaling if required
        w,h = kwargs['orig_size']
        w2,h2 = fig_size[0],fig_size[1]
        fig.set_size_inches([(w2/w)*w,(w2/w)*h])
        fig.set_dpi((w2/w)*fig.get_dpi())
    a=fig.gca()
    a.set_frame_on(False)
    a.set_xticks([]); a.set_yticks([])
    plt.axis('off')
    plt.xlim(0,h); plt.ylim(w,0)
    fig.savefig(fileName, transparent=True, bbox_inches='tight', \
                        pad_inches=0)

def roipoly(rowsize,colsize,xcontour,ycontour):
    xcontour = xcontour.reshape((-1,))
    ycontour = ycontour.reshape((-1,))
    contour = np.concatenate([xcontour.reshape((-1,1)), ycontour.reshape((-1,1))], axis = -1)
    #polyvet = np.zeros((2*xcontour.shape[0],))
    #polyvet[0::2] = ycontour
    #polyvet[1::2] = xcontour
    img = np.zeros(( rowsize, colsize))
    #ImageDraw.Draw(img).polygon(polyvet,outline=1,fill =1)
    #return np.array(img)
    cv2.fillPoly(img, np.int32([contour]),1)
    #cv2.fillPoly(img, contour, 1)
    return img

def fill_image(rowsize, colsize, contour_mat):
    filled_img  = np.zeros([temprow,tempcol])
    numCell = len(contour_mat)
    for icontour in range(0, numCell):
        thiscontour = contour_mat[icontour]
        xcontour = np.reshape(thiscontour[0,:].astype(int), (-1,1))
        ycontour = np.reshape(thiscontour[1,:].astype(int), (-1,1))
        #center_x = int(np.mean(xcontour))
        #center_y = int(np.mean(ycontour))
        #seed_map[center_y,center_x] = 1
        tempmask = roipoly(temprow,tempcol,xcontour, ycontour)
        filled_img = np.logical_or(filled_img, tempmask)
    return filled_img


def get_distance_for(seedCollection, temprow, tempcol, kind=1):
    '''get distance from the k-th index from the seedcollection to
    every pixel'''
    [tmpmeshRowInd, tmpmeshColomnInd] = np.meshgrid(range(0,tempcol),range(0,temprow))
    tmprealmaskind = np.concatenate((tmpmeshRowInd.flatten().reshape(-1,1),\
    tmpmeshColomnInd.flatten().reshape(-1,1)),axis =1);
    _, D = knnsearch(seedCollection,tmprealmaskind,kind)
    if myobj.decayparam:
      #realvalue_ = 1/(1+self.decalratio*D)
      realvalue_ = (np.exp(myobj.decayparam['alpha'] *(1- (D)/myobj.decayparam['r']))-1)/ (np.exp(myobj.decayparam['alpha']) -1)
      #realvalue_m1 = (1-np.exp(self.decayparam['alpha'] *((D)/self.decayparam['r']) - 1))/ (np.exp(self.decayparam['alpha']) -1)
    else:
      realvalue_ = D
    realvalue_[(realvalue_ <= 0).flatten()] = 0
    realvalue_[(filled_img==0).flatten()] = 0
    mask = realvalue_.reshape((temprow,tempcol))

def to_one_hot(indices, maxlen):
    if type(indices) in [int, float]:
       indices = [int(indices)]
    return np.asarray(np.eye(label_indx + 1)[indices])

def RGB2GRAY(rgb):
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

def RGB2YUV(input):
    R, G, B= input[...,0],input[...,1],input[...,2]

    Y = (0.299 * R + 0.587 * G + 0.114 * B)
    U = (-0.147 * R + -0.289 * G + 0.436 * B)
    V = (0.615 * R + -0.515 * G + -0.100 * B)
    return np.stack([Y, U, V], axis = -1).astype(int)

def YUV2RGB(input):
    Y, U, V= input[...,0],input[...,1],input[...,2]
    R = (Y + 1.14 * V)
    G = (Y - 0.39 * U - 0.58 * V)
    B = (Y + 2.03 * U)
    return np.stack([R, G, B], axis = -1).astype(int)

def imresize(img, resizeratio=1):
    '''Take care of cv2 reshape squeeze behevaior'''
    if resizeratio == 1:
        return img
    outshape = ( int(img.shape[1] * resizeratio) , int(img.shape[0] * resizeratio))
    temp = cv2.resize(img, outshape).astype(float)
    if len(img.shape) == 3 and img.shape[2] == 1:
        temp = np.reshape(temp, temp.shape + (1,))
    return temp

def imresize_shape(img, outshape):

    outshape = ( int(outshape[1]) , int(outshape[0])  )
    if (img.shape[0], img.shape[1]) == outshape:
        return img
    temp = cv2.resize(img, outshape).astype(float)
    if len(img.shape) == 3 and img.shape[2] == 1:
        temp = np.reshape(temp, temp.shape + (1,))
    return temp

def pre_process_img(img, yuv = False, mode = 'rescale', norm = True):
    if yuv :
      img = RGB2YUV(img)
    img = img.astype('float32')
    if len(img.shape) == 2:
        img = np.expand_dims(img,axis = -1)
    # first of all normalize the img
    if norm:
        if mode == 'channel':
            for ch_id in range(img.shape[2]):
                this_ch = img[:,:, ch_id]
                img[:,:, ch_id] = (this_ch - np.mean(this_ch)) / (np.std(this_ch) + 1e-8)
        elif mode == 'naive':
            img = img/(np.max(img[:]) + 1e-8)
        elif mode == 'rescale':
            img = (img - np.min(img[:])) / ( np.max(img[:]) - np.min(img[:]) + 1e-8)
            img = img - np.mean(img[:])
        else:
            raise Exception('Unknown mode for pre_processing')
    return img

def mysqueeze(a, axis = None):
    if axis == None:
        return np.squeeze(a)
    if a.shape[axis] != 1:
        return a
    else:
        return np.squeeze(a, axis = axis)

def getImg_from_Grid(grid_vec, patchsize):
    patchRow, patchCol = patchsize
    indx =  -1
    imgchannel = int(grid_vec.shape[-1]//(patchRow*patchCol))
    numberofImg = grid_vec.shape[0]
    gridshape = (grid_vec[0,:,:,:].shape[0],grid_vec[0,:,:,:].shape[1])
    imgs = np.zeros((grid_vec.shape[0], gridshape[0]*patchRow, gridshape[1]*patchCol, imgchannel ))
    imgs = mysqueeze(imgs, axis = -1)

    for imgidx  in range(numberofImg):
        for colid in range(gridshape[1]):
           for rowid in range(gridshape[0]):
              indx = indx + 1
              this_vec =  grid_vec[imgidx,rowid,colid,:]
              this_patch = np.reshape(this_vec, (patchRow,patchCol,imgchannel ))
              this_patch = mysqueeze(this_patch,axis = -1)
              startRow, endRow = rowid *patchRow, (rowid+1)*patchRow
              startCol, endCol = colid *patchCol, (colid+1)*patchCol
              #print this_patch.shape
              imgs[imgidx,startRow:endRow,startCol: endCol] = this_patch
              #imshow(img)
    return imgs

def getmesh_zigzag(RowPts,ColPts):
    """RowPts means the index of row coordinates,
       ColPts means the index of col coordinates
    """
    #img = np.zeros((max(RowPts), max(ColPts)))
    rr,cc = [], []
    for colidx in ColPts:
        if np.mod(colidx, 2) == 0:
           rr.extend(RowPts)
        else:
           rr.extend(RowPts[::-1])
        cc.extend([colidx]*len(RowPts))

    return np.asarray(rr), np.asarray(cc)

def getmesh(RowPts,ColPts):
    """RowPts means the index of row coordinates,
       ColPts means the index of col coordinates
    """
    rr,cc = [], []
    for colidx in ColPts:
        rr.extend(RowPts)
        cc.extend([colidx]*len(RowPts))
    return np.asarray(rr), np.asarray(cc)

def getfileinfo(imgdir, contourextList,ImgExtList,LabelExt):
    '''return a list of dictionary {'thisfile':os.path.join(imgdir,f), 'thismatfile':thismatfile}
    '''
    alllist  = [f for f in os.listdir(imgdir)]
    #absfilelist = [];
    #absmatfilelist = [];
    #returnDict = {}
    returnList = []
    for f in alllist:
        if os.path.isfile(os.path.join(imgdir,f)) and \
                   os.path.splitext(f)[1] in ImgExtList:
           flag = 0
           for contourext in contourextList:
             thismatfile  = os.path.join(imgdir,os.path.splitext(f)[0] + \
             contourext + LabelExt)
             if os.path.isfile(thismatfile):
                #absmatfilelist.append(thismatfile)
                #absfilelist.append(os.path.join(imgdir,f))
                #returnDict['thisfile'] = os.path.join(imgdir,f)
                #returnDict['thismatfile'] = thismatfile
                returnList.append({'thisfile':os.path.join(imgdir,f), 'thismatfile':thismatfile})
                #yield returnDict
                flag = 1
                break
           if flag == 0:
              print "Image: {s} does not have matfile".format(s = os.path.splitext(f)[0] )
    return returnList

def yieldfileinfo(imgdir, contourextList,ImgExtList,LabelExt):
    alllist  = [f for f in os.listdir(imgdir)]
    #absfilelist = [];
    #absmatfilelist = [];
    returnDict = {}
    returnList = []
    for f in alllist:
        if os.path.isfile(os.path.join(imgdir,f)) and \
                   os.path.splitext(f)[1] in ImgExtList:
           flag = 0
           for contourext in contourextList:
             thismatfile  = os.path.join(imgdir,os.path.splitext(f)[0] + \
             contourext + LabelExt)
             if os.path.isfile(thismatfile):
                #absmatfilelist.append(thismatfile)
                #absfilelist.append(os.path.join(imgdir,f))
                returnDict['thisfile'] = os.path.join(imgdir,f)
                returnDict['thismatfile'] = thismatfile
                #returnList.append({'thisfile':os.path.join(imgdir,f), 'thismatfile':thismatfile})
                yield returnDict
                flag = 1
                break
           if flag == 0:
              print "Image: {s} does not have matfile".format(s = os.path.splitext(f)[0] )

def getFromFolderList(subfolder_list,  number_list = -1, contourextList = '',
               ImgExtList = '.png',LabelExt = '.mat'):
    '''
    subfolder_list: the folder that contain the images,  it is a list of folders.
    number_list: the number of images you wanna take
    '''
    random.seed(4)
    if type(subfolder_list) != list:
        subfolder_list = [subfolder_list]
    if type(number_list) != list:
        number_list = [number_list]
    if len(number_list) == 1:
        number_list = number_list * len(subfolder_list)

    returnDict_list = []
    for imgdir, num_img in zip(subfolder_list, number_list):
        alllist  = [f for f in os.listdir(imgdir)]
        if len(subfolder_list) == 1 and len(alllist) < num_img:
            return None
            # because we want to terminate when the number of image is larger than this.

        total_num = len(alllist)
        list_of_file = range(total_num)
        random.shuffle(list_of_file)
        img_count = 0
        for file_ind in list_of_file:
            returnDict = {}
            f = alllist[file_ind]
            if os.path.isfile(os.path.join(imgdir,f)) and \
                    os.path.splitext(f)[1] in ImgExtList:
                flag = 0
                for contourext in contourextList:
                    thismatfile  = os.path.join(imgdir,os.path.splitext(f)[0] + \
                    contourext + LabelExt)
                    if os.path.isfile(thismatfile):
                        returnDict['thisfile'] = os.path.join(imgdir,f)
                        returnDict['thismatfile'] = thismatfile
                        returnDict_list.append(returnDict)
                        flag = 1
                        img_count += 1
                        break
                if flag == 0:
                    print "Image: {s} does not have matfile".format(s = os.path.splitext(f)[0] )
            if num_img > 0 and img_count == num_img:
                break
    return  returnDict_list

def getfilelist(Imagefolder, inputext):
    '''inputext: ['.json'] '''
    if type(inputext) is not list:
        inputext = [inputext]
    filelist = []
    filenames = []
    for f in os.listdir(Imagefolder):
        if os.path.splitext(f)[1] in inputext and os.path.isfile(os.path.join(Imagefolder,f)):
               filelist.append(os.path.join(Imagefolder,f))
               filenames.append(os.path.splitext(os.path.basename(f))[0])
    return filelist, filenames


def find(logicalMatrix):
    totalInd = np.arange(0, len(logicalMatrix.flat))
    return totalInd[logicalMatrix.flatten()]

def imshow(img, size=None):
    if size is not None:
        plt.figure(figsize = size)
    else:
        plt.figure()
    plt.imshow(img)
    plt.show()


def fast_Points2Patches(Patches,centerIndx, img, patchsize):
    totalsub = np.unravel_index(centerIndx, [img.shape[0],img.shape[1]])
    numberofInd = len(centerIndx)
    #Patches = np.zeros(numberofInd, np.prod(patchsize)*img.shape[2])
    if len(img.shape) == 2:
        img = img[:,:,None]
    npad3 = ((patchsize[0],patchsize[0]),(patchsize[1],patchsize[1]),(0,0))
    img = np.pad(img,npad3, 'symmetric')
    centralRow = totalsub[0][:] + patchsize[0]
    centralCol = totalsub[1][:] + patchsize[1]

    se = CentralToOrigin(centralRow, centralCol,patchsize[0],patchsize[1])

    for i in range(numberofInd):
        Patches[i,:] = img[se['RS'][i] : se['RE'][i], se['CS'][i]:se['CE'][i],:].flatten()


def knnsearch(seeds, pints,K):
    """return the indexes and distance of k neareast points for every pts in points from seeds\
    seeds: N*dim, points: N*dim
    seeds, and points should be of N*dim format"""
    knn = NearestNeighbors(n_neighbors=K)
    knn.fit(seeds)
    distance, index  = knn.kneighbors(pints, return_distance=True)
    return index,distance

def Points2Patches(centerIndx, img, patchsize):
    totalsub = np.unravel_index(centerIndx, [img.shape[0],img.shape[1]])
    numberofInd = len(centerIndx)
    if len(img.shape) == 2:
        img = img[:,:,None]
    Patches = np.zeros((numberofInd, np.prod(patchsize)*img.shape[2]))
    npad3 = ((patchsize[0],patchsize[0]),(patchsize[1],patchsize[1]),(0,0))
    img = np.pad(img,npad3, 'symmetric')
    centralRow = totalsub[0][:] + patchsize[0]
    centralCol = totalsub[1][:] + patchsize[1]

    se = CentralToOrigin(centralRow, centralCol,patchsize[0],patchsize[1])

    for i in range(numberofInd):
       Patches[i,:] = img[se['RS'][i] : se['RE'][i], se['CS'][i]:se['CE'][i],:].flatten()
            #imshow(img[se['RS'][i] : se['RE'][i], se['CS'][i]:se['CE'][i],:][...,0])
        #       tmp = img[:,:,0].copy() #np.zeros((img.shape[0], img.shape[1]))
        #       tmp[se['RS'][i] : se['RE'][i], se['CS'][i]:se['CE'][i]] = 255
        #       #tmp = scipy.ndimage.morphology.grey_dilation(tmp,(3,3) )
        #       imshow(tmp)
    return Patches


def CentralToOrigin(centralRow, centralCol,Rowsize,Colsize):

    RowUp = int(Rowsize/2)
    RowDown = Rowsize - RowUp - 1
    ColLeft = int(Colsize/2)
    ColRight = Colsize - ColLeft - 1
    se = {}
    se['RS'] = centralRow - RowUp
    se['RE'] = centralRow + RowDown + 1  #because python does not take the last value
    se['CS'] = centralCol - ColLeft
    se['CE'] = centralCol + ColRight   + 1
    return se

def OriginToCentral(OrigRow, OrigCol,Rowsize,Colsize):
    RowUp = int(Rowsize/2)
    ColLeft = int(Colsize/2)
    center = {}
    center['RC'] = OrigRow + RowUp
    center['CC'] = OrigCol + ColLeft
    return center


def patchflow(Img,chunknum,row,col,channel,**kwargs):

    pixelind = find(np.ones(Img.shape[0], Img.shape[1]) == 1)
    Totalnum = len(pixelind)
    numberofchunk = np.floor((Totalnum + chunknum - 1)// chunknum)   # the floor
    Chunkfile = np.zeros((chunknum, row*col*channel))

    chunkstart = 0
    for chunkidx in range(numberofchunk):
        thisnum = min(chunknum, Totalnum - chunkidx*chunknum)
        thisInd = pixelind[chunkstart: chunkstart + thisnum]
        fast_Points2Patches(Chunkfile[0:thisnum,:],thisInd, Img, (row,col))
        chunkstart += thisnum
        yield Chunkfile[0:thisnum,:]


def Indxflow(Totalnum, batch_size):
    numberofchunk = int(Totalnum + batch_size - 1)// int(batch_size)   # the floor
    #Chunkfile = np.zeros((batch_size, row*col*channel))
    totalIndx = np.random.permutation(np.arange(Totalnum))

    chunkstart = 0
    for chunkidx in range(int(numberofchunk)):
        thisnum = min(batch_size, Totalnum - chunkidx*batch_size)
        thisInd = totalIndx[chunkstart: chunkstart + thisnum]
        chunkstart += thisnum
        yield thisInd


def dataflow_two(BatchData, BatchMask,BatchBounary, batch_size):

    Totalnum = BatchData.shape[0]
    for thisInd in Indxflow(Totalnum, batch_size):
        yield BatchData[thisInd, ...], BatchMask[thisInd,...], BatchBounary[thisInd, ...]

def batchflow(batch_size, *Data):
    # we dont check Data, they should all have equal first dimension
    Totalnum = Data[0].shape[0]
    for thisInd in Indxflow(Totalnum, batch_size):
        if len(Data) == 1:
            yield Data[0][thisInd, ...]
        else:
            batch_tuple = [s[thisInd,...] for s in Data]
            yield tuple(batch_tuple)

def dataflow(BatchData, BatchLabel, batch_size):
    Totalnum = BatchData.shape[0]
    for thisInd in Indxflow(Totalnum, batch_size):
        yield BatchData[thisInd, ...], BatchLabel[thisInd, ...]


def overlayImg(img, mask,print_color =[5,119,72],linewidth= 1, alpha = 0.618,savepath = None):
    #img = img_as_float(data.camera())
    rows, cols = img.shape[0:2]
    # Construct a colour image to superimpose
    color_mask = np.zeros((rows, cols, 3))
    assert len(mask.shape) == 2,'mask should be of dimension 2'
    color_mask[mask == 1] = print_color
    color_mask[mask == 0] = img[mask == 0]
    #imshow(color_mask)

    if len(img.shape) == 2:
       img_color = np.dstack((img, img, img))
    else:
       img_color = img

    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)

    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    img_masked = color.hsv2rgb(img_hsv)
    # Display the output
    #f, (ax0, ax1, ax2) = plt.subplots(1, 3,
    #                                  subplot_kw={'xticks': [], 'yticks': []})
    #ax0.imshow(img, cmap=plt.cm.gray)
    #ax1.imshow(color_mask)
    #ax2.imshow(img_masked)
    #plt.show()

    img_masked = np.asarray((img_masked/np.max(img_masked) ) * 255, dtype = np.uint8)

    if savepath is not None:
        im = Image.fromarray(img_masked)
        im.save(savepath)

    #im = Image.fromarray(img_masked)
    #im.save(savepath)
    return img_masked

def mask2contour(org, mask, **kwargs):
    '''org are original reference image to be overlaid.
       mask should be binary image, anyway I will modify it using 100 as threshold to binarize it.

       if org, and mask are file names, then we read them
       if mask is numpy array, then  0 at background, 1 at foreground.

       savepath: full path to save the image
       kwargs:
            color
            linewidth
       org = '/home/yuanpuxie/Desktop/testImg/107_TAD_55.bmp'
       mask = '/home/yuanpuxie/Desktop/testImg/107_TAD_55.bmp.png'  #it should be a binary imag
       savepath = '/home/yuanpuxie/Desktop/testImg/save.png'
       mask2contour(org, mask, color = [0,0,1], linewidth=3)
    '''
    if isinstance(org, basestring):
        org = np.asarray(Image.open(org))

    if isinstance(mask, basestring):
        mask = np.asarray(Image.open(mask))
        print mask.shape
        if len(mask.shape) == 3:
            mask = mask[...,0]
            mask = (mask > 100)
    mask = mask.astype(int)
    contour_mask  = np.zeros(mask.shape)

    param = myobj()
    param.linewidth = 2
    param.color = [0,0,1] #[5,119,72]

    for key in kwargs:
        setattr(param, key, kwargs[key])

    contours = measure.find_contours(mask, 0)

    for n, contour in enumerate(contours):
        contour = contour.astype(int)
        #plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
        contour_mask[contour[:, 0], contour[:, 1]] = 1

    # dialte the image based on linewidth
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(param.linewidth,param.linewidth))
    contour_mask = cv2.dilate(contour_mask,se)

    return overlayImg(org, contour_mask , print_color = param.color, linewidth = 1, alpha = 1)

from skimage.measure import regionprops

def label2contour(org, label_img, print_color = [0,0,1], linewidth = 2, alpha = 1):
    npad = ((1,1),(1,1))
    label_img = np.pad(label_img, npad, mode='constant', constant_values=0)
    contour_img = np.zeros_like(label_img)
    tmp_img = np.zeros_like(label_img)
    regions = regionprops(label_img)
    region_count = 1
    #contourlist = []
    for props in regions[1:]:
        minr, minc, maxr, maxc = props.bbox
        tmp_img[label_img ==  props.label] = 1
        thispatch = tmp_img[minr-1:maxr+1, minc-1:maxc+1].copy()
        contours = measure.find_contours(thispatch, 0)
        thiscontour = (contours[0] + [minr-1, minc-1]).astype(int)
        #contourlist.append(thiscontour)
        contour_img[thiscontour[:, 0], thiscontour[:, 1]] = 1
        tmp_img.fill(0)

    contour_img = contour_img.astype(np.uint8)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(linewidth,linewidth))
    contour_mask = cv2.dilate(contour_img,se)[1:-1,1:-1]
    return overlayImg(org, contour_mask , print_color = print_color, alpha = alpha)


def split_img(img, windowsize=1000, board = 0):
    '''
    img dimension: channel, row, col
    output:
       (IndexDict, PackList)
       IndexDict is a dictionry, the key is the actual patch size, the values is the list of identifier,
       PackList: list of (thisPatch,org_slice ,extract_slice, thisSize,identifier), the index of Packlist
       corresponds to the identifier.
    eg:

    lenght = 12
    img = np.arange(2*lenght*lenght).reshape(2,lenght,lenght)

    nm = np.zeros(img.shape)

    AllDict, PackList =  split_img(img, windowsize=3, board = 2)

    print img

    print '---------------------------------------'

    print AllDict.keys()

    for key in AllDict.keys():
        iden_list = AllDict[key]
        for iden in iden_list:
            thispatch = PackList[iden][0]
            org_slice = PackList[iden][1]
            extract_slice = PackList[iden][2]

            nm[:,org_slice[0],org_slice[1]] = thispatch[:,extract_slice[0],extract_slice[1]]
    print nm

    '''

    if type(windowsize) is int:
        windowsize = (windowsize, windowsize)

    IndexDict = {}
    identifier = -1
    PackList = []

    numRowblocks = int(math.ceil(float(img.shape[1])/windowsize[0]))
    numColblocks = int(math.ceil(float(img.shape[2])/windowsize[1]))

    pad4d = ((0,0),(board,board), (board,board))
    pad_img = np.pad(img, pad4d, 'symmetric')

    thisrowstart, thiscolstart =0, 0
    thisrowend, thiscolend = 0,0
    for row_idx in range(numRowblocks):
        thisrowlen = min(windowsize[0], img.shape[1]-  row_idx * windowsize[0])

        if row_idx == 0:
           thisrowstart = 0
        else:
           thisrowstart = thisrowend

        thisrowend = thisrowstart + thisrowlen

        for col_idx in range(numColblocks):
            thiscollen = min(windowsize[1], img.shape[2]-  col_idx * windowsize[1])
            if col_idx == 0:
               thiscolstart = 0
            else:
               thiscolstart = thiscolend
            thiscolend = thiscolstart + thiscollen


            crop_r_start = thisrowstart - board + board
            crop_c_start = thiscolstart - board + board
            crop_r_end  = thisrowend + board + board
            crop_c_end  = thiscolend + board + board

            #we need to handle the tricky board condition

            # thispatch will be of size (:,:, windowsize+ 2*board)
            thisPatch =  pad_img[:,crop_r_start:crop_r_end, crop_c_start:crop_c_end]
            thisSize = (thisrowlen + 2*board , thiscollen + 2*board)

            org_slice = (slice(thisrowstart, thisrowend), slice(thiscolstart, thiscolend))

            extract_slice = (slice(board, board + thisrowlen), slice(board, board + thiscollen))

            identifier =  identifier +1
            PackList.append((thisPatch,org_slice ,extract_slice, thisSize,identifier))

            if thisSize in IndexDict:
               IndexDict[thisSize].append(identifier)
            else:
               IndexDict[thisSize] = []
               IndexDict[thisSize].append(identifier)

    return (IndexDict, PackList)
