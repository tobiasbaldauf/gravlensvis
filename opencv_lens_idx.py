import numpy as np
import matplotlib.pyplot as mpl
import cv2
import time


#grid setup
nclens=140
ncsource=720#240
deflgrid=np.zeros((nclens,nclens,2))
tgrid=np.zeros((nclens,nclens))
sgrid=np.zeros((ncsource,ncsource,3))
sgridg=np.zeros((ncsource,ncsource,3))
mapidx=np.zeros((nclens,nclens),dtype=np.int)
mapidx0=np.zeros((nclens,nclens),dtype=np.int)


'''

s   source webcam/gaussian image
m   erase all lenses
g   toggle grid
c   toggle color quadrants

left click  add point lens
right click move Gaussian source to cursor position

'''


mlist=[];

#parameters
dls=100.0
dl=100.0
ds=dl+dls
wl=1.0
ws=3.0
eps=10.0**-6
bfac=1
winsize=600

#grid spacings
dxsource=ws/ncsource
dxlens=wl/nclens

def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode

    if event == cv2.EVENT_LBUTTONDOWN:
        ix,iy = x,y
        cv2.circle(final_resize,(x,y),3,(0,0,255),-1)

    elif event == cv2.EVENT_LBUTTONUP:
        ix=np.mod(x,winsize)/(1.0*winsize)-0.5
        iy=y/(1.0*winsize)-0.5
        print('lens',ix,iy)
        pointdefl(ix,iy,0.001)
        calcimgidx(1)#mapidx=
        #mapidx0=calcimgidx(0)

        cv2.circle(final_resize,(x,y),3,(0,0,255),-1)

    elif event == cv2.EVENT_RBUTTONUP:
        ix=np.mod(x,winsize)/(1.0*winsize)-0.5
        iy=y/(1.0*winsize)-0.5
        print('source',iy*ws/wl,ix*ws/wl)
 #       pointdefl(ix,iy,0.001)
        gaussimg(iy*ds/dl,ix*ds/dl,ws/30.0)#sgridg=
#        calcimgidx(1)#mapidx=
        #mapidx0=calcimgidx(0)

        cv2.circle(final_resize,(x,y),3,(0,0,255),-1)


def gaussimg(x0,y0,sigma):
    #sgridg=np.zeros((ncsource,ncsource,3))
    for i in np.arange(ncsource):
        for j in np.arange(ncsource):
            x=(i-ncsource/2)*dxsource
            y=(j-ncsource/2)*dxsource
            sgridg[i,j,0]=1.0*np.exp(-1.0/2.0*((x-x0)**2.0+(y-y0)**2.0)/sigma**2.0)
    return 0#sgridg

#gaussimg(0.2,0.2,ws/50.0)


#deflection field produced by point mass
def pointdefl(x0,y0,mass):
    mlist.append([x0,y0])
    for i in np.arange(nclens):
        for j in np.arange(nclens):
            x=(i-nclens/2)*dxlens-y0
            y=(j-nclens/2)*dxlens-x0
            deflgrid[i,j,:]*=len(mlist)/(len(mlist)+1.0)
            deflgrid[i,j,0]-=mass/(len(mlist)+1.0)*x/np.sqrt(x**2.0+y**2.0+eps)**2.0
            deflgrid[i,j,1]-=mass/(len(mlist)+1.0)*y/np.sqrt(x**2.0+y**2.0+eps)**2.0


#interpolate source field at rays hit point
def interpimg(xhit,yhit):
    if (xhit>-ws/2.0 and xhit<ws/2.0 and yhit>-ws/2.0 and yhit<ws/2.0):
        ihit=np.int(np.floor(xhit/dxsource)+ncsource/2)
        jhit=np.int(np.floor(yhit/dxsource)+ncsource/2)
        return sgrid[ihit,jhit]
    else:
        return 0

#interpolate source field at rays hit point
def interpimgidx(xhit,yhit):
    if (xhit>-ws/2.0 and xhit<ws/2.0 and yhit>-ws/2.0 and yhit<ws/2.0):
        ihit=np.int(np.floor(xhit/dxsource)+ncsource/2)
        jhit=np.int(np.floor(yhit/dxsource)+ncsource/2)
        return isource[ihit,jhit]
    else:
        return 0


#calculate the image
def calcimg(defl):
    imgrid=np.zeros((nclens,nclens))
    for i in np.arange(nclens):
        for j in np.arange(nclens):
            x0=(i-nclens/2)*dxlens
            y0=(j-nclens/2)*dxlens
            xs=x0+dls*(x0/dl+deflgrid[i,j,0]*defl)
            ys=y0+dls*(y0/dl+deflgrid[i,j,1]*defl)
            imgrid[i,j]=interpimg(xs,ys)
    return imgrid

isource=np.arange(ncsource**2)
isource=np.reshape(isource,(ncsource,ncsource))

def interpimgidx(xhit,yhit):
    if (xhit>-ws/2.0 and xhit<ws/2.0 and yhit>-ws/2.0 and yhit<ws/2.0):
        ihit=np.int(np.floor(xhit/dxsource)+ncsource/2)
        jhit=np.int(np.floor(yhit/dxsource)+ncsource/2)
        return isource[ihit,jhit]
    else:
        return 0
    
def calcimgidx(defl):
    imgridid=np.zeros((nclens,nclens),dtype=np.int)
    for i in np.arange(nclens):
        for j in np.arange(nclens):
            x0=(i-nclens/2)*dxlens
            y0=(j-nclens/2)*dxlens
            xs=x0+dls*(x0/dl+deflgrid[i,j,0]*defl)
            ys=y0+dls*(y0/dl+deflgrid[i,j,1]*defl)
            dx=dls*(x0/dl+deflgrid[i,j,0]*defl)
            dy=dls*(y0/dl+deflgrid[i,j,1]*defl)
            tgrid[i,j]=(np.sqrt(x0**2.0+y0**2.0+dl**2.0)+np.sqrt(dx**2.0+dy**2.0+dls**2.0))/np.sqrt(x0**2.0+y0**2.0+dl**2.0)/ds*dl
            imgridid[i,j]=interpimgidx(xs,ys)
            mapidx[i,j]=imgridid[i,j]
            xs=x0+dls*(x0/dl+deflgrid[i,j,0]*0)
            ys=y0+dls*(y0/dl+deflgrid[i,j,1]*0)
            imgridid[i,j]=interpimgidx(xs,ys)
            mapidx0[i,j]=imgridid[i,j]
    #return imgridid



#reset deflection field
deflgrid=np.zeros((nclens,nclens,2))
mlist=[];
#pointdefl(0.2,0.2,0.001)
#pointdefl(-0.2,0.2,0.001)
pointdefl(0.0,0.0,0.001)
calcimgidx(1)#mapidx=
#calcimgidx(0)#mapidx0=


#setup webcam capture
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
iframe=0


dogrid=False
docol=False
srccam=True
gaussimg(0.0,0.0,ws/30.0)#sgridg=
#start = time.time()
while(True):
    start=time.time()
    #Capture frame-by-frame
    fx=cap.get(3);
    fy=cap.get(4);
    #cap.set(3, fx/4);
    #cap.set(4, fy/4);
    ret, frame = cap.read()
    #cv2.resize(frame, frame,(640, 360), 0, 0, cv2.INTER_CUBIC)
    iframe+=1
    # Our operations on the frame come here
    fx=cap.get(3);
    fy=cap.get(4);
    #print(fx,fy)
    if srccam:
        sgrid =frame[0:720,np.int(fx)/2-ncsource/2:np.int(fx)/2+ncsource/2,:]# cv2.cvtColor(frame[0:720,np.int(fx)/2-ncsource/2:np.int(fx)/2+ncsource/2], cv2.COLOR_BGR2GRAY)
        if docol:
            sgrid[0:ncsource/2,0:ncsource/2,0:2]=0.0
            sgrid[0:ncsource/2,ncsource/2:,1:3]=0.0
            sgrid[ncsource/2:,ncsource/2:,0]=0.0
            sgrid[ncsource/2:,ncsource/2:,2]=0.0
            sgrid[ncsource/2:,0:ncsource/2,1]=0.0
        if dogrid:
            ig=np.arange(ncsource,step=40)
            #print(ig)
            sgrid[ig,:]=0.0
            sgrid[:,ig]=0.0
            sgrid[ig+1,:]=0.0
            sgrid[:,ig+1]=0.0
            sgrid[ig-1,:]=0.0
            sgrid[:,ig-1]=0.0
    else:
        sgrid=sgridg
    

    sgridflat=np.reshape(sgrid,(ncsource**2,3))
    imgrid=np.reshape(sgridflat[np.reshape(mapidx,nclens**2)],(nclens,nclens,3))
    imgrid0=np.reshape(sgridflat[np.reshape(mapidx0,nclens**2)],(nclens,nclens,3))
    #imgrid=calcimg(1)
    #print(np.shape(imgrid))
    #imgrid0=calcimg(0)
    
   
    #Display the resulting frame
    font=cv2.FONT_HERSHEY_SIMPLEX
    if docol:
        final_frame1 = cv2.hconcat([imgrid0/bfac,imgrid/bfac])
    else:
        final_frame1 = cv2.hconcat([imgrid0/bfac,imgrid/bfac])
    
    #small = cv2.resize(final_frame1, (0,0), fx=5, fy=5)
    final_resize = cv2.resize(final_frame1, (winsize*2,winsize),cv2.INTER_CUBIC)
    cv2.putText(final_resize,'unlensed',(np.int(0.1*winsize),np.int(0.9*winsize)),font,1.3,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(final_resize,'lensed',(np.int(1.1*winsize),np.int(0.9*winsize)),font,1.3,(255,255,255),1,cv2.LINE_AA)
    #final_resize = cv2.cvtColor(final_resize2,cv2.COLOR_GRAY2RGB)
    #print(mlist)
    for a in mlist:
        cv2.circle(final_resize,(np.int((a[0]/dxlens+nclens/2)*winsize/nclens),np.int((a[1]/dxlens+nclens/2)*winsize/nclens)),8,(0,0,255),-1)
        cv2.circle(final_resize,(np.int((a[0]/dxlens+nclens/2)*winsize/nclens)+winsize,np.int((a[1]/dxlens+nclens/2)*winsize/nclens)),8,(0,0,255),-1)
    #cv2.cvNamedWindow("frame", CV_WINDOW_NORMAL);
    #cv2.cvSetWindowProperty('frame', CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
    #cvShowImage("Name", your_image);
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    #cv2.namedWindow('frame', cv2.WND_PROP_FULLSCREEN)
    #cv2.setWindowProperty('frame',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)#
    cv2.imshow('frame',final_resize)
    #cv2.setWindowProperty('frame',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)#
    #cv2.imshow('frame',frame)
    #cv2.setWindowProperty('frame',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback('frame',draw_circle)
    end = time.time()
    #print(end - start)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif cv2.waitKey(1) & 0xFF == ord('m'):
        deflgrid=np.zeros((nclens,nclens,2))
        mlist=[];
        calcimgidx(1)

    elif cv2.waitKey(1) & 0xFF == ord('s'):
        srccam= not srccam
    elif cv2.waitKey(1) & 0xFF == ord('c'):
        docol= not docol
    elif cv2.waitKey(1) & 0xFF == ord('g'):
        dogrid= not dogrid







#When everything done, release the capture
cap.release()
cv2.destroyAllWindows()