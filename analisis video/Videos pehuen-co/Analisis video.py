import cv2
import numpy as np
#import pandas as pd



#Importar video

fileName='0116_0040_18_11_01_10_33.avi'
cap = cv2.VideoCapture(fileName)

cantidadFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

font = cv2.FONT_HERSHEY_SIMPLEX

## convert your array into a dataframe
#df = pd.DataFrame (np.zeros([1]), columns=list('A'))

#por que se usa un tipo de video sin ningun tipo de estabilizacion, 
#el codigo funciona mejor con una cantidad muy limitada de frames


skipframes=5
#saltea una cantidad x de frames al inicio


#numero de frames que saltea al inicio
while(skipframes):
    cap.read()
    skipframes-=1
    
    
while(1):
    # Take first frame and find corners in it
    #ret, old_frame = cap.read()

    
    #Capturamos una imagen y la convertimos de RGB -> HSV
    ret,frame = cap.read()
    
    if 0==ret:#frame nulo
        break
            
    mat=np.matrix([[6.03693   ,2.75988    ,-1752  ],
         [0.02295   ,8.3054     ,-1877  ],
         [-.00002   ,0.00719    ,1      ]])
    
    tra =cv2.warpPerspective(frame,mat,(900,400))
    frame = tra
   # frame=frame[int(frame.shape[0]*1.5/4):int(frame.shape[0]*3/4),0:frame.shape[1]]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
 
    #Establecemos el rango de colores que vamos a detectar
    #En este caso de verde oscuro a verde-azulado claro
    blanco_bajos_hsv = np.array([80,0,180], dtype=np.uint8)
    blanco_altos_hsv = np.array([170,140,255], dtype=np.uint8)
    
    
    #Crear una mascara con solo los pixeles dentro del rango de verdes
    mask = cv2.inRange(hsv, blanco_bajos_hsv, blanco_altos_hsv)
    
    
#    Morphological Transformations
    
    
#    kernel = np.ones((3,10),np.uint8)
#    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    ###################################################   
    #Buscar las rectas que componen cada ola
    #Hough transform
    
    src=mask
    
    dst = cv2.Canny(src, 50, 200, None, 3)

    # Copy edges to the images that will display the results in BGR
    cdstP = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    sup = np.copy(frame)
    finales = np.copy(frame)
    alturaMarea = np.copy(frame)
    
    lineCont=0
    
    linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 100, None, 30, 10)
    
    if linesP is not None:   #Si no encuentra lineas con Hough proteje de errores

        angProm=np.zeros(len(linesP))
        angProm[:]=np.angle(linesP[:,0,2]-linesP[:,0,0]+linesP[:,0,3]*1j-linesP[:,0,1]*1j, deg=True)

        angMedian=np.median(angProm) 
        linesSave=np.zeros((linesP.shape))
        maxAltura=np.amax(linesP[:,0,1])

   
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
            cv2.line(sup, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)

            #dibuja las lineas que tienen un angulo proximo a la media de las lineas encontradas en total
            angle =np.angle(l[2]-l[0]+l[3]*1j-l[1]*1j, deg=True)
            if angle<=(angMedian+10) and angle>=(angMedian-10):
                cv2.line(sup, (l[0], l[1]), (l[2], l[3]), (0,255,0), 3, cv2.LINE_AA)
                cv2.line(finales, (l[0], l[1]), (l[2], l[3]), (0,255,0), 3, cv2.LINE_AA)
                
                if l[1]>maxAltura-20:       #deja solo las lineas de las olas mas cercanas a la costa
                    linesSave[lineCont]=linesP[i][0]
                    lineCont+1
                    
                    cv2.line(alturaMarea, (l[0], l[1]), (l[2], l[3]), (100,100,0), 3, cv2.LINE_AA)
                    
        #calcula el primedio de las lineas de costa
        alturaLinea=np.nanmedian(np.where(linesSave[:,0,1]>0,linesSave[:,0,1],np.NAN))
        
        cv2.putText(alturaMarea,str(alturaLinea) ,(30, 30),font,1,(0, 0, 255), thickness=2)
#        df2 = pd.DataFrame (linesDistanceWithpreviousLineFiltradas[linesDistanceWithpreviousLineFiltradas[:,0]!=0,0], columns=list('A'))
#        df=df.append(df2, ignore_index=True)
#        df=df.append(pd.DataFrame ([0], columns=list('A')), ignore_index=True)
        cv2.imshow("Source", src)
        cv2.imshow("Finales", finales)
        cv2.imshow("Altura Marea", alturaMarea)
        cv2.imshow("Superposicion", sup)
        cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
        cv2.imshow("transformada", tra)

        #cv2.imshow('mask HSV', mask)
        #cv2.imshow('mask RGB', mask2)
        #cv2.imshow('Frame', frame)




        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        
## convert your array into a dataframe

## save to xlsx file
#Guarda las longitudes de onda en un archivo xlsx


#filepath = 'longitudes de onda.xlsx'

#df.to_excel(filepath, index=False)

#cv2.imwrite('mask.png',mask)
cv2.destroyAllWindows()
cap.release()