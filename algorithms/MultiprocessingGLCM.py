import math
import numpy as np
from multiprocessing import Process, Queue

# Classe do MultiprocessingGLCM
class MultiprocessingGLCM:
    def __init__(self):
        self.subImages = None
        self.subImagesDescriptors = []
    
    def setSubImages(self, subImages):
        self.subImages = subImages

    def run(self):
        self._resetSubImagesDescriptors()
        self._distribute()

    def _resetSubImagesDescriptors(self):
        self.subImagesDescriptors = []

    def _calculateFeatures(self, matrix):
        entropy, energy, homogeneity, contrast, correlation = 0, 0, 0, 0, 0
        
        meanX = np.mean(matrix, axis=1)
        meanY = np.mean(matrix, axis=0)
        stdX = np.std(matrix, axis=1)
        stdY = np.std(matrix, axis=0)

        for x in range(0, 256):
            for y in range(0, 256):
                entropy += (-1 * matrix[x, y] * math.log(matrix[x, y])) if matrix[x, y] != 0 else 0
                energy += math.pow(matrix[x, y], 2)
                homogeneity += matrix[x, y] / (1 + abs(x - y))
                contrast += math.pow((x - y), 2) * matrix[x, y]
                correlation += ((((x - meanX[x]) * (y - meanY[y])) / (stdX[x] * stdY[y])) * matrix[x, y]) if stdX[x] != 0 and stdY[y] != 0 else 0

        return [entropy, energy, homogeneity, contrast, correlation]

    def _process(self, processID, subImage, queue):
        height, width = subImage.shape

        # 0 Degree
        aMatrix = np.zeros((256, 256))
        # 45 Degrees
        bMatrix = np.zeros((256, 256))
        # 90 Degrees
        cMatrix = np.zeros((256, 256))
        # 135 Degrees
        dMatrix = np.zeros((256, 256))
        
        # 0 Degree
        for x in range(0, height):
            for y in range(0, width - 1):
                aMatrix[subImage[x, y], subImage[x, y + 1]] += 1
                aMatrix[subImage[x, y + 1], subImage[x, y]] += 1 
        # 45 Degrees
        for x in range(1, height):
            for y in range(0, width - 1):
                bMatrix[subImage[x, y], subImage[x - 1, y + 1]] += 1
                bMatrix[subImage[x - 1, y + 1], subImage[x, y]] += 1
        # 90 Degrees
        for x in range(0, height - 1):
            for y in range(0, width):
                cMatrix[subImage[x, y], subImage[x + 1, y]] += 1
                cMatrix[subImage[x + 1, y], subImage[x, y]] += 1

        # 135 Degrees
        for x in range(0, height - 1):
            for y in range(0, width - 1):
                dMatrix[subImage[x, y], subImage[x + 1, y + 1]] += 1
                dMatrix[subImage[x + 1, y + 1], subImage[x, y]] += 1
        
        aMatrix /= (2 * height * (width - 1))
        bMatrix /= (2 * (height - 1) * (width - 1))
        cMatrix /= (2 * (height - 1) * width)
        dMatrix /= (2 * (height - 1) * (width - 1))
        
        descriptor = []
        descriptor.extend(self._calculateFeatures(aMatrix))
        descriptor.extend(self._calculateFeatures(bMatrix))
        descriptor.extend(self._calculateFeatures(cMatrix))
        descriptor.extend(self._calculateFeatures(dMatrix))
        
        queue.put({
            'processID': processID,
            'processDescriptor': descriptor
        })

    def _distribute(self):
        processes = []
        queue = Queue()

        for i, subImage in enumerate(self.subImages):
            process = Process(target=self._process, args=(i, subImage, queue))
            process.start()
            processes.append(process)

        descriptors = [queue.get() for process in processes]
        [process.join() for process in processes]

        descriptors = sorted(descriptors, key=lambda k: k['processID'])
        for descriptor in descriptors:
            self.subImagesDescriptors.append(descriptor['processDescriptor'])
