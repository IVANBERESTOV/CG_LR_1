import numpy as np
import math as m
from PIL import Image,ImageOps;

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
                polOne = [0] * (6)
                curPoint=2
                for i in range(0,len(s)):
                    if(i%3==0):
                        if(polOne[2]!=0):
                            pol.append(polOne)
                        polOne[0]=int(s[1].split('/')[0])
                        polOne[3]=int(s[1].split('/')[1])
                    else:
                        polOne[i%3]=int(s[curPoint].split('/')[0])
                        polOne[i%3+3]=int(s[curPoint].split('/')[1])
                        curPoint+=1
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

def paint(matrix, zbuf, cameraCoef, sizeCoef, ver, pol, normArr, vt, txtres=np.array(0)):
    if(pol!=0):
        for i in pol:
            temp=cut(ver[i[0]-1][0],ver[i[0]-1][1],ver[i[0]-1][2],ver[i[1]-1][0],ver[i[1]-1][1],ver[i[1]-1][2],ver[i[2]-1][0],ver[i[2]-1][1],ver[i[2]-1][2])
            if(temp<0):
                draw_tr(matrix, zbuf, cameraCoef, sizeCoef, ver[i[0]-1][0],ver[i[0]-1][1],ver[i[0]-1][2],ver[i[1]-1][0],ver[i[1]-1][1],ver[i[1]-1][2],ver[i[2]-1][0],ver[i[2]-1][1],ver[i[2]-1][2],normArr[i[0]-1],normArr[i[1]-1],normArr[i[2]-1],i,vt,txtres)

def getMatrix(H, W, color):
    matrix = np.zeros((W, H, 3), dtype=np.uint8)
    for i in range(0, W):
        for j in range(0, H):
            matrix[i][j] = color
    return matrix

def getZBuff(matrix):
    matrix = np.zeros((matrix.shape[0],matrix.shape[1]),dtype=np.float64)
    for i in range(0, matrix.shape[0]):
        for j in range(0, matrix.shape[1]):
            matrix[i][j]=np.inf
    return matrix

def saveMatrixAsIMG(matrix, name='image1.png'):
    image = Image.fromarray(matrix, mode='RGB')
    image = ImageOps.flip(image)
    image.save(name)

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

def draw_tr(matrix,zbuf,cameraCoef,sizeCoef,x0,y0,z0,x1,y1,z1,x2,y2,z2,i0,i1,i2,ipol,vt,txtres=np.array(0)):
    # Сюда добавляем
    a = (7000*cameraCoef)*sizeCoef
    px0, py0 =  (a * x0 / z0 + W/2), (a * y0 /z0 + H/2)
    px1, py1 =  (a * x1 / z1 + W/2), (a * y1 / z1 + H/2)
    px2, py2 =  (a * x2 / z2 + W/2), (a * y2 / z2 + H/2)
    
    I0=i0[2]
    I1=i1[2]
    I2=i2[2]

    p0ResCoord=(vt[ipol[3]-1][1], vt[ipol[3]-1][0])
    p1ResCoord=(vt[ipol[4]-1][1], vt[ipol[4]-1][0])
    p2ResCoord=(vt[ipol[5]-1][1], vt[ipol[5]-1][0])

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
                if(x<matrix.shape[1] and y<matrix.shape[0]):
                    if(z < zbuf[y][x]):
                        I = l0*I0+l1*I1+l2*I2
                        I = max(0, min(1,-I))
                        if(len(txtres.shape)!=0):
                            color = I * txtres[int((txtres.shape[0]-1)*(l0*p0ResCoord[0]+l1*p1ResCoord[0]+l2*p2ResCoord[0]))][int((txtres.shape[1]-1)*(l0*p0ResCoord[1]+l1*p1ResCoord[1]+l2*p2ResCoord[1]))]
                        else:
                            color = 255 * I
                        matrix[y][x] = color
                        zbuf[y][x] = z

def nor(x0,y0,z0,x1,y1,z1,x2,y2,z2):
    return np.cross(np.array([x1-x2,y1-y2,z1-z2]),np.array([x1-x0,y1-y0,z1-z0]))

def cut(x0,y0,z0,x1,y1,z1,x2,y2,z2):
    l=[0,0,1]
    var = np.dot(nor(x0,y0,z0,x1,y1,z1,x2,y2,z2),l)/(np.linalg.norm(nor(x0,y0,z0,x1,y1,z1,x2,y2,z2))*np.linalg.norm(l))
    return var

def quatMult(q1, q2):
    a1, b1, c1, d1 = q1
    a2, b2, c2, d2 = q2
    a = a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2
    b = a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2
    c = a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2
    d = a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2
    return np.array([a, b, c, d])

def quatNorm(q):
    norm = np.linalg.norm(q)
    return q / norm if norm > 0 else q

def resizeNew(ver, tx=0, ty=0, tz=0, quat_rotation=None):
    if quat_rotation is None:
        quat_rotation = np.array([1.0, 0.0, 0.0, 0.0])
    
    for i in ver:
        rotated = quatRotateVector(quat_rotation, [i[0], i[1], i[2]])
        i[0], i[1], i[2] = rotated[0] + tx, rotated[1] + ty, rotated[2] + tz
    return ver

def quatRotateVector(quater, vect):
    a, b, c, d = quater
    v_quater = np.array([0, vect[0], vect[1], vect[2]])
    q_c = np.array([a, -b, -c, -d])
    rotated = quatMult(quatMult(quater, v_quater), q_c)
    return rotated[1:]

def getQuat(alpha, beta, gamma):
    cx = np.cos(alpha * 0.5)
    sx = np.sin(alpha * 0.5)

    cy = np.cos(beta * 0.5)
    sy = np.sin(beta * 0.5)

    cz = np.cos(gamma * 0.5)
    sz = np.sin(gamma * 0.5)
    
    a = cx * cy * cz + sx * sy * sz
    b = sx * cy * cz - cx * sy * sz
    c = cx * sy * cz + sx * cy * sz
    d = cx * cy * sz - sx * sy * cz
    return quatNorm([a, b, c, d])

W = 1600
H = 900
matrixIMG = getMatrix(W, H,(0, 0, 0))
cameraCoefA=10.5
sizeCoefA=0.1
cameraCoefB=0.17
sizeCoefB=0.5
txtresA = np.array(ImageOps.flip(Image.open("AfroRes.bmp")))
verA, polA , vtA = objParser("Afro.obj")
normArrA = np.zeros((len(verA), 3))
verA=resizeNew(verA,-1,-0.5,cameraCoefA, getQuat(0, m.pi, 0))
normArrA=getNormArr(verA,polA,normArrA)
txtresB = np.array(ImageOps.flip(Image.open("BunnyRes.jpg")))
verB, polB , vtB = objParser("Bunny.obj")
normArrB = np.zeros((len(verB), 3))
verB=resizeNew(verB,0.1, -0.05,cameraCoefB, getQuat(0, -m.pi/2, 0))
normArrB=getNormArr(verB,polB,normArrB)
zbuf=getZBuff(matrixIMG)
paint(matrixIMG,zbuf,cameraCoefA,sizeCoefA,verA,polA,normArrA,vtA, txtresA)
#paint(matrixIMG,zbuf,cameraCoefA,sizeCoefA,verA,polA,normArrA,vtA)
paint(matrixIMG,zbuf,cameraCoefB,sizeCoefB,verB,polB,normArrB,vtB, txtresB)
saveMatrixAsIMG(matrixIMG)