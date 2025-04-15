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
    vt = []
    with open(file_str, 'r') as f:
        for line in f:
            s=line.split(' ')
            if(s[0] == 'v'):
                ver.append([float(x) for x in s[1:4]])
            if(s[0] == 'f'):
                polOne = [0] * 6
                for i in range(0,3):
                    polOne[i]=int(s[i+1].split('/')[0])
                for i in range(0,3):
                    polOne[i+3]=int(s[i+1].split('/')[1])
                pol.append(polOne)
            if(s[0] == 'vt'):
                vt.append([float(x) for x in s[1:3]])
    return ver, pol, vt

def getNormArr(ver, pol, normArr):
    for i in pol:
        temp =  nor(ver[i[0]-1][0],ver[i[0]-1][1],ver[i[0]-1][2],ver[i[1]-1][0],ver[i[1]-1][1],ver[i[1]-1][2],ver[i[2]-1][0],ver[i[2]-1][1],ver[i[2]-1][2])
        temp/=np.linalg.norm(temp)
        normArr[i[0]-1] += temp
        normArr[i[1]-1] += temp
        normArr[i[2]-1] += temp
    for j in range(0,len(normArr)):
        normArr[j] = normArr[j]/np.linalg.norm(normArr[j])
    return normArr

def paint(matrix, zbuf, ver, pol, normArr,vt):
    if(pol!=0):
        for i in pol:
            temp=cut(ver[i[0]-1][0],ver[i[0]-1][1],ver[i[0]-1][2],ver[i[1]-1][0],ver[i[1]-1][1],ver[i[1]-1][2],ver[i[2]-1][0],ver[i[2]-1][1],ver[i[2]-1][2])
            if(temp<0):
                draw_tr(matrix, zbuf, ver[i[0]-1][0],ver[i[0]-1][1],ver[i[0]-1][2],ver[i[1]-1][0],ver[i[1]-1][1],ver[i[1]-1][2],ver[i[2]-1][0],ver[i[2]-1][1],ver[i[2]-1][2],normArr[i[0]-1],normArr[i[1]-1],normArr[i[2]-1],i,vt)

def getMatrix(H, W, color):
    matrix = np.zeros((W, H, 3), dtype=np.uint8)
    for i in range(0, W):
        for j in range(0, H):
            matrix[i][j] = color
    return matrix

def getZBuff(H,W):
    matrix = np.zeros((W,H),dtype=np.float64)
    for i in range(0, W):
        for j in range(0, H):
            matrix[i][j]=np.inf
    return matrix

def saveMatrixAsIMG(matrix):
    image = Image.fromarray(matrix, mode='RGB')
    image = ImageOps.flip(image)
    image.save('image1.png')

def rotMatrix(alpha=np.radians(0),beta=np.radians(0),gamma=np.radians(0)):
    Rx=np.array(
        [[1,0,0],
        [0, np.cos(alpha),np.sin(alpha)],
        [0,-np.sin(alpha),np.cos(alpha)]]
    )

    Ry=np.array(
        [[np.cos(beta),0,np.sin(beta)],
        [0, 1,0],
        [-np.sin(beta),0,np.cos(beta)]]
    )

    Rz=np.array(
        [[np.cos(gamma), np.sin(gamma),0],
        [-np.sin(gamma),np.cos(gamma),0],
        [0, 0, 1]]
    )
    return np.dot(Rx,np.dot(Ry,Rz))

def resize(ver,tx=0,ty=0,tz=0,alpha=np.radians(0),beta=np.radians(0),gamma=np.radians(0)):
    for i in ver:
        i[0],i[1],i[2] = np.dot(rotMatrix(alpha,beta,gamma),[i[0], i[1], i[2]]) + [tx,ty,tz]
    return ver
   

def bary(x0,y0,x1,y1,x2,y2,x,y):
    l0=((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    l1=((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    l2 = 1.0 - l0 - l1
    return l0, l1, l2

def draw_tr(matrix,zbuf,x0,y0,z0,x1,y1,z1,x2,y2,z2,i0,i1,i2,ipol,vt):
    # Сюда добавляем
    a = 7000*coef
    px0, py0 =  a * x0 / z0 + W/2, a * y0 /z0 + H/2
    px1, py1 =  a * x1 / z1 + W/2, a * y1 / z1 + H/2
    px2, py2 =  a * x2 / z2 + W/2, a * y2 / z2 + H/2
    
    I0=i0[2]
    I1=i1[2]
    I2=i2[2]

    p0ResCoord=(vt[ipol[3]-1][1],vt[ipol[3]-1][0])
    p1ResCoord=(vt[ipol[4]-1][1],vt[ipol[4]-1][0])
    p2ResCoord=(vt[ipol[5]-1][1],vt[ipol[5]-1][0])

    # И везде дальше меняем x на px
    xmin=min(px0,px1,px2)
    xmax=max(px0,px1,px2)
    ymin=min(py0,py1,py2)
    ymax=max(py0,py1,py2)
    if(xmin<0):xmin=0
    if(ymin<0):ymin=0
    xmin=int(m.floor(xmin))
    xmax=int(m.ceil(xmax))
    ymin=int(m.floor(ymin))
    ymax=int(m.ceil(ymax))
    for x in range(xmin,xmax+1):
        for y in range(ymin,ymax+1):
            l0,l1,l2 = bary(px0,py0,px1,py1,px2,py2,x,y)
            if(l0>=0 and l1>=0 and l2>=0):
                z=l0*z0+l1*z1+l2*z2
                if(z < zbuf[y][x]):
                    I = l0*I0+l1*I1+l2*I2
                    color = (-I) * txtres[int(txtres.shape[0]*(l0*p0ResCoord[0]+l1*p1ResCoord[0]+l2*p2ResCoord[0]))][int(txtres.shape[1]*(l0*p0ResCoord[1]+l1*p1ResCoord[1]+l2*p2ResCoord[1]))]
                    #color*=-225
                    matrix[y][x] = color
                    zbuf[y][x] = z

def nor(x0,y0,z0,x1,y1,z1,x2,y2,z2):
    return np.cross(np.array([x1-x2,y1-y2,z1-z2]),np.array([x1-x0,y1-y0,z1-z0]))

def cut(x0,y0,z0,x1,y1,z1,x2,y2,z2):
    l=[0,0,1]
    var = np.dot(nor(x0,y0,z0,x1,y1,z1,x2,y2,z2),l)/(np.linalg.norm(nor(x0,y0,z0,x1,y1,z1,x2,y2,z2))*np.linalg.norm(l))
    return var

W = 1600
H = 900
coef=0.17
txtres = np.array(ImageOps.flip(Image.open("bunny-atlas.jpg")))
print(txtres.shape)
matrix = getMatrix(W, H,(0, 0, 0))
ver, pol , vt = objParser("model_1.obj")
normArr = np.zeros((len(ver), 3))
ver=resize(ver,0,-0.05,1*coef,np.radians(0),np.radians(0),np.radians(0))
normArr=getNormArr(ver,pol,normArr)
zbuf=getZBuff(W,H)
paint(matrix,zbuf,ver,pol,normArr,vt)
saveMatrixAsIMG(matrix)