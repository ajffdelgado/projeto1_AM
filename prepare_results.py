# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 10:22:27 2018

@author: Delgado
"""

acuracia_media =[76.66666666666667, 79.52380952380952, 76.19047619047619, 81.9047619047619, 78.0952380952381, 82.85714285714286, 78.0952380952381, 81.9047619047619, 83.80952380952381, 75.23809523809524, 76.66666666666667, 79.52380952380952, 76.19047619047619, 81.9047619047619, 78.0952380952381, 82.85714285714286, 78.0952380952381, 81.9047619047619, 83.80952380952381, 75.23809523809524, 76.66666666666667, 79.52380952380952, 76.19047619047619, 81.9047619047619, 78.0952380952381, 82.85714285714286, 78.0952380952381, 81.9047619047619, 83.80952380952381, 75.23809523809524, 76.66666666666667, 79.52380952380952, 76.19047619047619, 81.9047619047619, 78.0952380952381, 82.85714285714286, 78.0952380952381, 81.9047619047619, 83.80952380952381, 75.23809523809524, 76.66666666666667, 79.52380952380952, 76.19047619047619, 81.9047619047619, 78.0952380952381, 82.85714285714286, 78.0952380952381, 81.9047619047619, 83.80952380952381, 75.23809523809524, 76.66666666666667, 79.52380952380952, 76.19047619047619, 81.9047619047619, 78.0952380952381, 82.85714285714286, 78.0952380952381, 81.9047619047619, 83.80952380952381, 75.23809523809524, 76.66666666666667, 79.52380952380952, 76.19047619047619, 81.9047619047619, 78.0952380952381, 82.85714285714286, 78.0952380952381, 81.9047619047619, 83.80952380952381, 75.23809523809524, 76.66666666666667, 79.52380952380952, 76.19047619047619, 81.9047619047619, 78.0952380952381, 82.85714285714286, 78.0952380952381, 81.9047619047619, 83.80952380952381, 75.23809523809524, 76.66666666666667, 79.52380952380952, 76.19047619047619, 81.9047619047619, 78.0952380952381, 82.85714285714286, 78.0952380952381, 81.9047619047619, 83.80952380952381, 75.23809523809524, 76.66666666666667, 79.52380952380952, 76.19047619047619, 81.9047619047619, 78.0952380952381, 82.85714285714286, 78.0952380952381, 81.9047619047619, 83.80952380952381, 75.23809523809524, 76.66666666666667, 79.52380952380952, 76.19047619047619, 81.9047619047619, 78.0952380952381, 82.85714285714286, 78.0952380952381, 81.9047619047619, 83.80952380952381, 75.23809523809524, 76.66666666666667, 79.52380952380952, 76.19047619047619, 81.9047619047619, 78.0952380952381, 82.85714285714286, 78.0952380952381, 81.9047619047619, 83.80952380952381, 75.23809523809524, 76.66666666666667, 79.52380952380952, 76.19047619047619, 81.9047619047619, 78.0952380952381, 82.85714285714286, 78.0952380952381, 81.9047619047619, 83.80952380952381, 75.23809523809524, 76.66666666666667, 79.52380952380952, 76.19047619047619, 81.9047619047619, 78.0952380952381, 82.85714285714286, 78.0952380952381, 81.9047619047619, 83.80952380952381, 75.23809523809524, 76.66666666666667, 79.52380952380952, 76.19047619047619, 81.9047619047619, 78.0952380952381, 82.85714285714286, 78.0952380952381, 81.9047619047619, 83.80952380952381, 75.23809523809524, 76.66666666666667, 79.52380952380952, 76.19047619047619, 81.9047619047619, 78.0952380952381, 82.85714285714286, 78.0952380952381, 81.9047619047619, 83.80952380952381, 75.23809523809524, 76.66666666666667, 79.52380952380952, 76.19047619047619, 81.9047619047619, 78.0952380952381, 82.85714285714286, 78.0952380952381, 81.9047619047619, 83.80952380952381, 75.23809523809524, 76.66666666666667, 79.52380952380952, 76.19047619047619, 81.9047619047619, 78.0952380952381, 82.85714285714286, 78.0952380952381, 81.9047619047619, 83.80952380952381, 75.23809523809524, 76.66666666666667, 79.52380952380952, 76.19047619047619, 81.9047619047619, 78.0952380952381, 82.85714285714286, 78.0952380952381, 81.9047619047619, 83.80952380952381, 75.23809523809524, 76.66666666666667, 79.52380952380952, 76.19047619047619, 81.9047619047619, 78.0952380952381, 82.85714285714286, 78.0952380952381, 81.9047619047619, 83.80952380952381, 75.23809523809524, 76.66666666666667, 79.52380952380952, 76.19047619047619, 81.9047619047619, 78.0952380952381, 82.85714285714286, 78.0952380952381, 81.9047619047619, 83.80952380952381, 75.23809523809524, 76.66666666666667, 79.52380952380952, 76.19047619047619, 81.9047619047619, 78.0952380952381, 82.85714285714286, 78.0952380952381, 81.9047619047619, 83.80952380952381, 75.23809523809524, 76.66666666666667, 79.52380952380952, 76.19047619047619, 81.9047619047619, 78.0952380952381, 82.85714285714286, 78.0952380952381, 81.9047619047619, 83.80952380952381, 75.23809523809524, 76.66666666666667, 79.52380952380952, 76.19047619047619, 81.9047619047619, 78.0952380952381, 82.85714285714286, 78.0952380952381, 81.9047619047619, 83.80952380952381, 75.23809523809524, 76.66666666666667, 79.52380952380952, 76.19047619047619, 81.9047619047619, 78.0952380952381, 82.85714285714286, 78.0952380952381, 81.9047619047619, 83.80952380952381, 75.23809523809524, 76.66666666666667, 79.52380952380952, 76.19047619047619, 81.9047619047619, 78.0952380952381, 82.85714285714286, 78.0952380952381, 81.9047619047619, 83.80952380952381, 75.23809523809524, 76.66666666666667, 79.52380952380952, 76.19047619047619, 81.9047619047619, 78.0952380952381, 82.85714285714286, 78.0952380952381, 81.9047619047619, 83.80952380952381, 75.23809523809524, 76.66666666666667, 79.52380952380952, 76.19047619047619, 81.9047619047619, 78.0952380952381, 82.85714285714286, 78.0952380952381, 81.9047619047619, 83.80952380952381, 75.23809523809524, 76.66666666666667, 79.52380952380952, 76.19047619047619, 81.9047619047619, 78.0952380952381, 82.85714285714286, 78.0952380952381, 81.9047619047619, 83.80952380952381, 75.23809523809524, 76.66666666666667, 79.52380952380952, 76.19047619047619, 81.9047619047619, 78.0952380952381, 82.85714285714286, 78.0952380952381, 81.9047619047619, 83.80952380952381, 75.23809523809524]
lista_final = []
lista_temp = []
for i in range(len(acuracia_media)):
    if (i %10 == 0 and i != 0):
        lista_final.append(lista_temp)
        lista_temp = []
        lista_temp.append(acuracia_media[i])
    else:
        lista_temp.append(acuracia_media[i])
lista_final.append(lista_temp)