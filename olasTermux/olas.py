# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 13:19:12 2019

@author: diego
"""

#Importar video
import cv2
import numpy as np

fileName='test.png'


font = cv2.FONT_HERSHEY_SIMPLEX

# Take first frame and find corners in it
#ret, old_frame = cap.read()
frame= cv2.imread(fileName)

        
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
 
#Establecemos el rango de colores que vamos a detectar
#En este caso de verde oscuro a verde-azulado claro
#blanco_bajos = np.array([200,200,200], dtype=np.uint8)
#blanco_altos = np.array([255, 255, 255], dtype=np.uint8)
#blanco_bajos_hsv = np.array([141,25,125], dtype=np.uint8)
#blanco_altos_hsv = np.array([162,140,255], dtype=np.uint8)
blanco_bajos_hsv = np.array([0,0,180], dtype=np.uint8)
blanco_altos_hsv = np.array([140,140,255], dtype=np.uint8)


#Crear una mascara con solo los pixeles dentro del rango de verdes
#mask2 = cv2.inRange(frame, blanco_bajos, blanco_altos)
mask = cv2.inRange(hsv, blanco_bajos_hsv, blanco_altos_hsv)

###################################################   
#Buscar las rectas que componen cada ola
#Hough transform

src=mask

dst = cv2.Canny(src, 50, 200, None, 3)

# Copy edges to the images that will display the results in BGR
cdstP = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
sup = np.copy(frame)
finales = np.copy(frame)

linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 30, 10)

if linesP is not None:   #Si no encuentra lineas con Hough proteje de errores

    angProm=np.zeros(len(linesP))
    angProm[:]=np.angle(linesP[:,0,2]-linesP[:,0,0]+linesP[:,0,3]*1j-linesP[:,0,1]*1j, deg=True)

    angMedian=np.median(angProm) 
    linesSave=np.zeros((linesP.shape))
    linesSaveExpanse=np.zeros((linesP.shape))
    linesDistanceWithpreviousLine=np.zeros([np.int(linesP[:,0,0].shape[0]),2])
    linesDistanceWithpreviousLineFiltradas=np.zeros([np.int(linesP[:,0,0].shape[0]),2])

    lineNew=np.zeros([2,2])
    lineOld=np.zeros([2,2])
    lineCont=0
    lineContF=0


    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
            cv2.line(sup, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)

            #dibuja las lineas que tienen un angulo proximo a la media de las lineas encontradas en total
            angle =np.angle(l[2]-l[0]+l[3]*1j-l[1]*1j, deg=True)
            if angle<=(angMedian+10) and angle>=(angMedian-10):
                cv2.line(sup, (l[0], l[1]), (l[2], l[3]), (0,255,0), 3, cv2.LINE_AA)
                linesSave[lineCont]=linesP[i][0]

                if np.tan(angMedian*np.pi/180)*(1080-linesSave[lineCont,0,0])+linesSave[lineCont,0,1]<=1920:
                    linesSaveExpanse[lineCont]=[linesSave[lineCont,0,0],linesSave[lineCont,0,1],1080,np.tan(angMedian*np.pi/180)*(1080-linesSave[lineCont,0,0])+linesSave[lineCont,0,1]]


                elif ((1920-linesSave[lineCont,0,1])/np.tan(angMedian*np.pi/180))+linesSave[lineCont,0,0]<=1080:

                    linesSaveExpanse[lineCont]=[linesSave[lineCont,0,0],linesSave[lineCont,0,1],((1920-linesSave[lineCont,0,1])/np.tan(angMedian*np.pi/180))+linesSave[lineCont,0,0],1920]

                linesSaveExpanse = linesSaveExpanse[(-linesSaveExpanse[:,0,3]).argsort()]

                if lineCont>=1:
                    lineNew[0]=[linesSaveExpanse[lineCont,0,0],linesSaveExpanse[lineCont,0,1]]
                    lineNew[1]=[linesSaveExpanse[lineCont,0,2],linesSaveExpanse[lineCont,0,3]]
                    lineOld[0]=[linesSaveExpanse[lineCont-1,0,0],linesSaveExpanse[lineCont-1,0,1]]
                    lineOld[1]=[linesSaveExpanse[lineCont-1,0,2],linesSaveExpanse[lineCont-1,0,3]]

                    linesDistanceWithpreviousLine[lineCont]=np.divide(np.cross([lineNew[0],lineNew[1]], [lineNew[1],lineOld[1]]),np.linalg.norm(lineNew))
                    #linesDistanceWithpreviousLine[lineCont]=-lineNew[1,1]+lineOld[1,1]

                    if linesDistanceWithpreviousLine[lineCont,1]!=linesDistanceWithpreviousLine[lineCont-1,1]:
                        if linesDistanceWithpreviousLine[lineCont,1]>5 or linesDistanceWithpreviousLine[lineCont,1]<-5:


                            cv2.line(finales, (np.int(lineNew[1,0]), np.int(lineNew[1,1])), (np.int(lineOld[1,0]), np.int(lineOld[1,1])), (255,0,0), 3, cv2.LINE_AA)

                            cv2.line(finales, (np.int(lineNew[0,0]), np.int(lineNew[0,1])), (np.int(lineNew[1,0]), np.int(lineNew[1,1])), (0,255,0), 3, cv2.LINE_AA)
                            cv2.line(finales, (np.int(lineOld[0,0]), np.int(lineOld[0,1])), (np.int(lineOld[1,0]), np.int(lineOld[1,1])), (0,255,0), 3, cv2.LINE_AA)

                            cv2.putText(finales,str(linesDistanceWithpreviousLine[lineCont,1]),(np.int(lineNew[1,0]), np.int(lineNew[1,1])), font, 0.75,(255,255,255),2,cv2.LINE_AA)

                            linesDistanceWithpreviousLineFiltradas[lineContF]=linesDistanceWithpreviousLine[lineCont]
                            lineContF+=1

                lineCont+=1

    #########buscar la distancia perpendicular a la recta ente las rectas, ignorando las rectas que se encuantrean a la misma distancia
    
    cv2.imwrite("Source.jpg", src)
    cv2.imwrite("Finales.jpg", finales)
    cv2.imwrite("Superposicion.jpg", sup)
    cv2.imwrite("Detected Lines (in red) - Probabilistic Line Transform.jpg", cdstP)

    #cv2.imshow('mask HSV', mask)
    #cv2.imshow('mask RGB', mask2)
    #cv2.imshow('Frame', frame)





        
## convert your array into a dataframe

## save to xlsx file
#Guarda las longitudes de onda en un archivo xlsx



#cv2.imwrite('mask.png',mask)