import numpy as np
from scipy import ndimage as ndi
import cv2
from skimage.filters import threshold_otsu
#morpho
import skimage.morphology as morpho 

#%%

def contrast_stretching (ima):
    im = ((ima - ima.min()) / (ima.max()-ima.min())) * 255
    return(im.astype('uint8'))

#%%
def histogram_equalization(image, number_bins=256):
    

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = (number_bins-1) * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)
    image_equalized = image_equalized.reshape(image.shape)
    return image_equalized.astype('uint8')
#%%
def RGBtoGRAYnumpy (ima, i = 0):
    
    'If i = 0 (default) Convert from RGB PIL to grayscale numpy'
    'If i= 1 convert from RGB numpy to grayscale numpy'
    
    if i==0:
        im = ima.convert('L')
        return np.array(im)
    
    elif i==1:
        r, g, b = ima[:,:,0], ima[:,:,1], ima[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        
    else:
        print('No valid parameter i: \n  If i = 0 (default) Convert from RGB PIL to grayscale numpy \n If i= 1 convert from RGB numpy to grayscale numpy')
        return
    return gray
#%%
def histogram(im):
    
    nl,nc=im.shape    
    hist=np.zeros(256)
    
    for i in range(nl):
        for j in range(nc):
            hist[im[i][j]]=hist[im[i][j]]+1
            
    for i in range(256):
        hist[i]=hist[i]/(nc*nl)
        
    return(hist)
#%%
def otsu_thresh3(im):
    
    h=histogram(im)
    
    m=0
    for i in range(256):
        m=m+i*h[i]
    
    maxt1=0
    maxt2=0   #modif
    maxk=0
    
    
    for t1 in range(256):
        for t2 in range(256):     #modif
            w0=0
            w1=0
            w2=0          #modif
            m0=0
            m1=0
            m2=0        #modif
            for i in range(t1):
                w0=w0+h[i]
                m0=m0+i*h[i]
            if w0 > 0:
                m0=m0/w0

            for i in range(t1,t2):     #modif
                w1=w1+h[i]
                m1=m1+i*h[i]
            if w1 > 0:   
                m1=m1/w1
            for i in range(t2,256):    #modif
                w2=w2+h[i]
                m2=m2+i*h[i]
            if w2 > 0:   
                m2=m2/w2

            k=w0*w1*w2*(m0-m1)*(m0-m1)*(m0-m2)*(m0-m2)*(m1-m2)*(m1-m2)

            if k > maxk:
                maxk=k
                maxt1=t1
                maxt2=t2
            
            
    thresh=(maxt1,maxt2)
        
    return(thresh)
#%%
def filtergauss(im):
    """applique un filtre passe-bas gaussien. coupe approximativement a f0/4"""
    (ty,tx)=im.shape
    imt=np.float32(im.copy())
    pi=np.pi
    XX=np.concatenate((np.arange(0,tx/2+1),np.arange(-tx/2+1,0)))
    XX=np.ones((ty,1))@(XX.reshape((1,tx)))
    
    YY=np.concatenate((np.arange(0,ty/2+1),np.arange(-ty/2+1,0)))
    YY=(YY.reshape((ty,1)))@np.ones((1,tx))
    # C'est une gaussienne, dont la moyenne est choisie de sorte que
    # l'integrale soit la meme que celle du filtre passe bas
    # (2*pi*sig^2=1/4*x*y (on a suppose que tx=ty))
    sig=(tx*ty)**0.5/2/(pi**0.5)
    mask=np.exp(-(XX**2+YY**2)/2/sig**2)
    imtf=np.fft.fft2(imt)*mask
    return np.real(np.fft.ifft2(imtf))

#%%
def median_filter(im,typ=1,r=1,xy=None):
    """ renvoie le median de l'image im.
    si typ==1 (defaut) le median est calcule sur un carre de cote 2r+1
    si typ==2 : disque de rayon r
    si typ==3 alors xy est un couple de liste de x et liste de y
         ([-1,0,1] , [0,0,0]) donne un median sur un segment horizontql de taille trois. 
         """
    lx=[]
    ly=[]
    (ty,tx)=im.shape
    if typ==1: #carre

        for k in range(-r,r+1):
            for l in range(-r,r+1):
                lx.append(k)
                ly.append(l)

    elif typ==2:
        for k in range(-r,r+1):
            for l in range(-r,r+1):
                if k**2+l**2<=r**2:
                    lx.append(k)
                    ly.append(l)
    else: #freeshape
        lx,ly=xy

    debx=-min(lx) #min is supposed negatif
    deby=-min(ly)
    finx=tx-max(lx) #max is supposed positif
    finy=ty-max(ly)
    ttx=finx-debx
    tty=finy-deby
    tab=np.zeros((len(lx),ttx*tty))
    #print (lx,ly)
    #print(ttx,tty)
    #print(im[deby+ly[k]:tty+ly[k]+deby,debx+lx[k]:debx+ttx+lx[k]].reshape(-1).shape)
    for k in range(len(lx)):
        tab[k,:]=im[deby+ly[k]:deby+tty+ly[k],debx+lx[k]:debx+ttx+lx[k]].reshape(-1)
    out=im.copy()
    out[deby:finy,debx:finx]=np.median(tab,axis=0).reshape((tty,ttx))
    return out

#%%

def otsu_thresh(im):

    h=histogram(im)

    m=0
    for i in range(256):
        m=m+i*h[i]

    maxt=0
    maxk=0


    for t in range(256):
        w0=0
        w1=0
        m0=0
        m1=0
        for i in range(t):
            w0=w0+h[i]
            m0=m0+i*h[i]
        if w0 > 0:
            m0=m0/w0

        for i in range(t,256):
            w1=w1+h[i]
            m1=m1+i*h[i]
        if w1 > 0:
            m1=m1/w1

        k=w0*w1*(m0-m1)*(m0-m1)

        if k > maxk:
            maxk=k
            maxt=t


    thresh=maxt

    return(thresh)

#%%
def filtre_lineaire(im,mask):
    """ renvoie la convolution de l'image avec le mask. Le calcul se fait en 
utilisant la transformee de Fourier et est donc circulaire.  Fonctionne seulement pour 
les images en niveau de gris.
"""
    fft2=np.fft.fft2
    ifft2=np.fft.ifft2
    (y,x)=im.shape
    (ym,xm)=mask.shape
    mm=np.zeros((y,x))
    mm[:ym,:xm]=mask
    fout=(fft2(im)*fft2(mm))
    # on fait une translation pour ne pas avoir de decalage de l'image
    # pour un mask de taille impair ce sera parfait, sinon, il y a toujours un decalage de 1/2
    mm[:ym,:xm]=0
    y2=int(np.round(ym/2-0.5))
    x2=int(np.round(xm/2-0.5))
    mm[y2,x2]=1
    out=np.real(ifft2(fout*np.conj(fft2(mm))))
    return out

#%%
def preprocessing(ima, p=0):
    ima_pre = contrast_stretching(ima)
    
    if p==0:
        ima_pre = histogram_equalization(ima_pre)
        
    ima_pre = ndi.median_filter(ima_pre, 2)

    return ima_pre

#%%
def HR_blackhat(ima):
    # Kernel for the morphological filtering
    kernel = cv2.getStructuringElement(1,(17,17)) #structuring element cross
    
    # Perform the blackHat filtering on the grayscale image to find the hair countours
    blackhat = cv2.morphologyEx(ima, cv2.MORPH_BLACKHAT, kernel)

    a, thrBH = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY) #binary threshold with threshold=10
    
    thresh = otsu_thresh(ima)
    binary = ima > thresh
    
    img1 = binary*255 + thrBH #We remove the hair and imperfections from the foreground.
    
    thresh = threshold_otsu(img1)       #OTSU threshhold of the mask with hair remo
    binary = img1 > thresh
    
    footprint_disk= morpho.disk(10) #structuring element disk of radius 10
    out = morpho.binary_opening(binary, footprint_disk) #opening 
    
    
    return out*255   #truncate to int values (not bool)

#%%
def Hair_removal(mask, thrBH):
    img = mask*255 + thrBH #We remove the hair and imperfections from the foreground.
    
    thresh = threshold_otsu(img)       #OTSU threshhold of the mask with hair remo
    print(thresh)
    
    footprint_disk = morpho.disk(10) #structuring element disk of radius 10
    
    return(img > thresh)
def Bord_remove(ima):
    p=0
    
    pad = cv2.copyMakeBorder(ima, 1,1,1,1, cv2.BORDER_CONSTANT, value=0) #Introduction of a black frame width 1
     
    h, w = pad.shape
    
    # create zeros mask 2 pixels larger in each dimension
    mask = np.zeros([h + 2, w + 2], np.uint8)
    
    img_floodfill = cv2.floodFill(pad, mask, (0,0), 255, (5), (0), flags=8)[1] #Truncate to 255 all the elements connected to (0,0)
    
    img_floodfill = img_floodfill[1:(h-1), 1:(w-1)] #reshape to original size (border added pixels)
        
    if ((np.count_nonzero(img_floodfill)/(h*w))>0.9):  
        p = 1

    return p, img_floodfill
        
#%%
def postprocessing(ima):
    ## CLOSING
    footprint_disk = morpho.disk(50) #structuring element disk of radius 30
    image_closing_disk= morpho.binary_closing(ima, footprint_disk) #closing 
    
    ## OPENING
    fin = morpho.binary_opening(image_closing_disk, footprint_disk)
    
    return fin

#%%
def seg_method(ima):
    p=0
    ima_p = preprocessing(ima, p) #Image conditioning
    
    ima_HR = HR_blackhat(ima_p)  #Hair removal
    
    a, ima_BR = Bord_remove(ima_HR) #Bord remove
    
    if a==1:
        ima_p = preprocessing(ima, a) #Image conditioning
        ima_HR = HR_blackhat(ima_p)  #Hair removal
        a, ima_BR = Bord_remove(ima_HR) #Bord remove
        
    out = postprocessing(ima_BR)
    return np.invert(out)
    
    
    
    
    
    

