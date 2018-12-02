# @author: Luan Utimura
# Processamento de Imagens Digitais - PPGCC 2018

import os, sys
import argparse
import operator
import numpy as np
import scipy.spatial as sp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tqdm import tqdm
from algorithms import *

# Variável global para armazenar a quantidade de
# classes (depende do tamanho da sub-imagem [parâmetro --w]);
numOfClasses = 0

# Função utilizada para criar um vetor com o nome dos arquivos
# da base de dados Brodatz;
def getFilenames(numOfImages):
    return ['D{}.gif'.format(i) for i in range(1, numOfImages + 1)]

# Função utilizada para criar um vetor com as sub-imagens de uma
# dada imagem;
def getSubImages(filename, windowSize):
    originalImg = mpimg.imread(filename)
    
    subImg = []

    for x in range(0, originalImg.shape[0] - windowSize + 1, windowSize):
        for y in range(0, originalImg.shape[1] - windowSize + 1, windowSize):
            subImg.append(originalImg[x:x+windowSize, y:y+windowSize])
    return subImg

# Função utilizada para calcular a distância de uma amostra ("base")
# em relação à todas as outras (com a func. euclidiana);
def getRelativeED(base, descriptors):
    relativeDistances = {}

    for i in range(0, descriptors.shape[0]):
        if i == base: continue
        relativeDistances[i] = sp.distance.euclidean(descriptors[i], descriptors[base])

    return relativeDistances

# Função utilizada para calcular o vetor de PR de uma amostra ("i")
# levando em consideração seu vetor de distâncias relativas calculado
# anteriormente;
def getDescriptorPR(i, sortedRelativeDistances): 
    global numOfClasses

    currentDescriptorClass = i // numOfClasses
        
    retrievedInClass = 0
    totalRetrieved = 0
    totalInClass = numOfClasses - 1
        
    currentDescriptorPR = [0] * (numOfClasses - 1)
    for descriptorNumber, relativeDistance in sortedRelativeDistances:
        totalRetrieved += 1
        if (descriptorNumber // numOfClasses) == currentDescriptorClass:
            retrievedInClass += 1
            currentDescriptorPR[retrievedInClass - 1] = retrievedInClass / totalRetrieved
    return currentDescriptorPR

if __name__ == '__main__':
	# Realiza o tratamento dos parâmetros passados durante a execução
    parser = argparse.ArgumentParser(description='Calculates the avg. AUC-PR of a given method (LBP|GLCM|WLD) over the Brodatz\'s dataset.')
    parser.add_argument('--a', dest='algorithm', type=str, default='lbp', help='Algorithm to use: <lbp|glcm|wld>')
    parser.add_argument('--w', dest='window', type=int, default=320, help='Window size to use: <320|160|...>')
    arguments = parser.parse_args()
    
    # Calcula a quantidade de classes (ou neste caso, amostras),
    # dado o tamanho que uma sub-imagem deve possuir;
    numOfClasses = (640 // arguments.window) * (640 // arguments.window)
    print('Number of classes: {}'.format(numOfClasses))

    algorithms = {
        'lbp': MultiprocessingLBP.MultiprocessingLBP,
        'glcm': MultiprocessingGLCM.MultiprocessingGLCM,
        'wld': MultiprocessingWLD.MultiprocessingWLD
    }

    if arguments.algorithm not in algorithms:
        print('Invalid algorithm: {}.'.format(arguments.algorithm))
        sys.exit(1)

    algorithmClass = algorithms[arguments.algorithm]
    
    descriptors = []
    alg = algorithmClass()

    # Para cada imagem da base de dados Brodatz,
    # executa o algoritmo do descritor escolhido.
    # (Note que são passadas todas as sub-imagens de uma vez.
    #  Por ser multiprocessing, os descritores são calculados paralelamente.)
    imgFilenames = getFilenames(112)
    for imgFilename in tqdm(imgFilenames):
        subImages = getSubImages(imgFilename, arguments.window)

        alg.setSubImages(subImages)
        alg.run()
        
        descriptors.extend(alg.subImagesDescriptors)
    
    # Realiza a normalização dos descritores GLCM e WLD.
    # (Divide pelo máximo de cada coluna);
    descriptors = np.array(descriptors)
    if arguments.algorithm in ['glcm', 'wld']:
        columnsMax = descriptors.max(axis=0)
        
        for i in range(len(columnsMax)):
            if columnsMax[i] == 0:
                columnsMax[i] = 1
        descriptors = descriptors / columnsMax
    
    # Realiza o cálculo do vetor PR de cada uma das amostras e,
    # consequentemente, calcula a média deles.
    descriptorsPR = []
    for i in range(0, descriptors.shape[0]):
        relativeDistances = getRelativeED(i, descriptors)
        sortedRelativeDistances = sorted(relativeDistances.items(), key=operator.itemgetter(1))
        currentDescriptorPR = getDescriptorPR(i, sortedRelativeDistances) 
        descriptorsPR.append(currentDescriptorPR)

    descriptorsPR = np.array(descriptorsPR)
    descriptorsPR_mean = np.mean(descriptorsPR, axis=0)
    
    print('{}| Average Precision x Recall:'.format(arguments.algorithm.upper()))
    print(descriptorsPR_mean)
