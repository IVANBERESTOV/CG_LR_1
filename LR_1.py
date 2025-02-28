import numpy as np
import math as m
from PIL import Image,ImageOps;

def dotted_line(matrix, x0, y0, x1, y1, count, color):
    step=1.0/count
    for t in np.arange(0, 1, step):
        x=round((1.0-t)*x0+t*x1)
        y=round((1.0-t)*y0+t*y1)
        matrix[y,x]=color

def dotted_line_v2(matrix, x0, y0, x1, y1, color):
    count = m.sqrt((x0-x1)**2+(y0-y1)**2)
    step=1.0/count
    for t in np.arange(0, 1, step):
        x=round((1.0-t)*x0+t*x1)
        y=round((1.0-t)*y0+t*y1)
        matrix[y,x]=color

def x_loop_line(matrix, x0, y0, x1, y1, color):
    for x in range(x0,x1):
        t = (x-x0)/(x1-x0)
        y = round((1.0-t)*y+t*y1)
        matrix[y,x] = color

def x_loop_line_fix1(matrix, x0, y0, x1, y1, color):
    if(x0>x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    for x in range(x0,x1):
        t = (x-x0)/(x1-x0)
        y = round((1.0-t)*y+t*y1)
        matrix[y,x] = color

def x_loop_line_fix2(matrix, x0, y0, x1, y1, color):
    flag=False
    if(abs(x0-x1)<abs(y0-y1)):
        x0,y0=y0,x0
        x1,y1=y1,x1
        flag = True
    if(x0>x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    for x in range(x0,x1):
        t = (x-x0)/(x1-x0)
        y = round((1.0-t)*y+t*y1)
        if(flag):
            matrix[x,y] = color
        else:
            matrix[y,x] = color

def Brithenham(matrix, x0, y0, x1, y1, color, shadow=0):
    flag=False
    if(abs(x0-x1)<abs(y0-y1)):
        x0,y0=y0,x0
        x1,y1=y1,x1
        flag = True
    if(x0>x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    y=y0
    dy=2*abs(y1-y0)
    derror = 0.0
    y_update = 1 if y1>y0 else -1
    for x in range(x0,x1):
        t = (x-x0)/(x1-x0)
        #y = round((1.0-t)*y+t*y1)
        if(flag):
            matrix[x,y] = color if(not shadow) else shadow
        else:
            matrix[y,x] = color if(not shadow) else shadow
        derror+=dy
        if(derror>(x1-x0)):
            derror-=2*(x1-x0)
            y+=y_update
    
def Stars(matrix):
    for i in range(0,13):
        x0=100
        y0=100
        x1=int(100+95*np.cos(2*i*np.pi/13))
        y1=int(100+95*np.sin(2*i*np.pi/13))
        Brithenham(matrix,x0,y0,x1,y1,(255,3,3))

def objParser(file_str):
    ver = []
    pol = []
    with open(file_str, 'r') as f:
        for line in f:
            s=line.split(' ')
            if(s[0] == 'v'):
                ver.append([float(x) for x in s[1:4]])
            if(s[0] == 'f'):
                polOne = [0]*3
                for i in range(0,3):
                    polOne[i]=int(s[i+1].split('/')[0])
                pol.append(polOne)
    return ver, pol

def paint(matrix, color, ver, pol=0,shadow=False):
    for i in ver:
        x=int(i[0])
        y=int(i[1])
        matrix[y][x]=color
    if(pol!=0):
        for i in pol:
            temp = [0] * 2
            for j in range(0,4):
                temp[1]=i[j%3]
                if(temp[0]):Brithenham(matrix, int(ver[temp[0]-1][0]), int(ver[temp[0]-1][1]),int(ver[temp[1]-1][0]),int(ver[temp[1]-1][1]), color if (not shadow)else ver[temp[0]-1][2])
                temp[0]=temp[1]

def getMatrix(H, W, color):
    matrix = np.zeros((W, H, 3), dtype=np.uint8)
    for i in range(0, W):
        for j in range(0, H):
            matrix[i][j] = color
    return matrix

def saveMatrixAsIMG(matrix):
    image = Image.fromarray(matrix, mode='RGB')
    image = ImageOps.flip(image)
    image.save('image1.png')

def resize(W, H, ver, zShadow=False, color=0, shadowIntens=0):
    xMax,yMax,zMax,xMin,yMin,zMin = ver[0][0],ver[0][1],ver[0][2],ver[0][0],ver[0][1],ver[0][2]
    for i in ver:
        if(i[0]>xMax):xMax=i[0]
        if(i[0]<xMin):xMin=i[0]
        if(i[1]>yMax):yMax=i[1]
        if(i[1]<yMin):yMin=i[1]
        if(i[2]>zMax):zMax=i[2]
        if(i[2]<zMin):zMin=i[2]
    absX=abs(xMax-xMin)
    absY=abs(yMax-yMin)
    absZ=abs(zMax-zMin)
    if(absX<0.0001):absX=0.0001
    if(absY<0.0001):absY=0.0001
    if(absZ<0.0001):absZ=0.0001

    coefX=int((W-10)/absX)
    coefY=int((H-10)/absY)
    
    cenX=int(coefX*(xMax+xMin)/2)
    cenY=int(coefY*(yMax+yMin)/2)
    cenW=W/2
    cenH=H/2
    pointsToCenterW=cenW-cenX
    pointsToCenterH=cenH-cenY
    
    
    if(zShadow):
        shadowIntens
        coefZR=int((color[0]-int(color[0]/shadowIntens))/absZ)
        coefZG=int((color[1]-int(color[1]/shadowIntens))/absZ)
        coefZB=int((color[2]-int(color[2]/shadowIntens))/absZ)
        pointsToZeroZR=int(color[0]/shadowIntens)-int(zMin*coefZR)
        pointsToZeroZG=int(color[1]/shadowIntens)-int(zMin*coefZG)
        pointsToZeroZB=int(color[2]/shadowIntens)-int(zMin*coefZB)
        for i in ver:
            i[0],i[1],i[2] = i[0]*coefX+pointsToCenterW,i[1]*coefY+pointsToCenterH,(int(i[2]*coefZR+pointsToZeroZR),int(i[2]*coefZG+pointsToZeroZG),int(i[2]*coefZB+pointsToZeroZB))
    else:
        for i in ver:
            i[0],i[1] = i[0]*coefX+pointsToCenterW,i[1]*coefY+pointsToCenterH
    return ver


W = 1600
H = 900
matrix = getMatrix(W, H,(0, 0, 0))
ver, pol = objParser("model_1.obj")
ver=resize(W, H, ver,True,(170,255,100),50)
paint(matrix, (255,255,255), ver, pol, True)
saveMatrixAsIMG(matrix)