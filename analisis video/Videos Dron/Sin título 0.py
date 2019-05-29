# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 13:24:52 2018

@author: diego
"""

import cv2
import numpy as np
####Funciones
def buscarMasCercano( mask, memoria, fila, columna, puntos, contadorPuntos ):
    for xfila in range(-5,5):
        for xcolumna in range(-5,5):
            if memoria[fila+xfila, columna+xcolumna]==1:
                memoria[fila+xfila, columna+xcolumna]=0

                if mask[fila, columna,0]==255:
                    contadorPuntos+=1
                    puntos[contadorPuntos]=fila,columna
                    buscarMasCercano( mask, memoria, fila+xfila, columna+xcolumna, puntos, contadorPuntos )

    return

frame=cv2.imread("mask.png")
mask=cv2.imread("mask.png")

#cv2.imshow('mask HSV', mask)

filas, columnas =frame[:,:,0].shape
memoria=np.zeros(frame[:,:,0].shape)
memoria[:,:]=1
puntos=np.zeros([3000,2])
rectasPuntos=np.zeros([5000,3000,2])
#frameRectas=np.zeros([cantidadFrames,1000,3000,2])
contadorPuntos=0
contadorRectas=0

##recorro la mascara de luminancia para hayar las rectas de las olas
for fila in range (0, filas):
    for columna in range (0, columnas):
        if memoria[fila, columna]==1:
            memoria[fila, columna]=0
            if mask[fila, columna,0]==255:

                contadorPuntos=0
                puntos[:]=0,0
                puntos[contadorPuntos]=fila,columna

                ##busco el siguiente punto mas cercano en un radio de 10 pixeles
                buscarMasCercano( mask, memoria, fila, columna, puntos, contadorPuntos)
                rectasPuntos[contadorRectas]=puntos
                contadorRectas+=1


#frameRectas[actualFrame]=rectasPuntos
import xlsxwriter

workbook = xlsxwriter.Workbook('arrays.xlsx')
worksheet = workbook.add_worksheet()


f, c =rectasPuntos[:,:,0].shape

for row in range(0,f):
    for col in range(0,c):
        worksheet.write(row, col, np.int(rectasPuntos[row,col,0]))

workbook.close()