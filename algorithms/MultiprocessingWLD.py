import math
import scipy.signal
import numpy as np

from multiprocessing import Process, Queue

# Classe do MultiprocessingWLD;
# (A implementação desta classe seguiu o artigo do Moodle e em partes o artigo
# "Face recognition using Weber local descriptors" Shutao Li, Dayi Gong, Yuan Yuan);
class MultiprocessingWLD:
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

    def _mapTheta(self, v3, v4, theta):
        if v3 > 0 and v4 > 0:
            return theta
        elif v3 > 0 and v4 < 0:
            return theta + (2 * math.pi)
        else:
            return theta + math.pi
    
    def _quantizedT1(self, T1, alpha):
        interval = [(-0.5 * math.pi, -0.3 * math.pi), (-0.3 * math.pi, -0.15 * math.pi), (-0.15 * math.pi, -0.05 * math.pi),
         (-0.05 * math.pi, 0), (0, 0.05 * math.pi), (0.05 * math.pi, 0.15 * math.pi), (0.15 * math.pi, 0.25 * math.pi),
         (0.25 * math.pi, 0.5 * math.pi)]
        
        for i in range(T1):
            if i < (T1 - 1):
                if alpha >= interval[i][0] and alpha < interval[i][1]:
                    return i
            else:
                if alpha >= interval[i][0] and alpha <= interval[i][1]:
                    return i

    def _quantizedT2(self, T2, mappedTheta):
        interval = [(0, 0.15 * math.pi), (0.15 * math.pi, 0.35 * math.pi), (0.35 * math.pi, 0.5 * math.pi),
                (0.5 * math.pi, 0.65 * math.pi), (0.65 * math.pi, 0.85 * math.pi), (0.85 * math.pi, 1.0 * math.pi),
                (1.0 * math.pi, 1.15 * math.pi), (1.15 * math.pi, 1.35 * math.pi), (1.35 * math.pi, 1.5 * math.pi),
                (1.5 * math.pi, 1.65 * math.pi), (1.65 * math.pi, 1.85 * math.pi), (1.85 * math.pi, 2.0 * math.pi)]

        for i in range(T2):
            if i < (T2 - 1):
                if mappedTheta >= interval[i][0] and mappedTheta < interval[i][1]:    
                    return i
            else:
                if mappedTheta >= interval[i][0] and mappedTheta < interval[i][1]:
                    return i

    def _process(self, processID, subImage, queue):
        height, width = subImage.shape
        
        f1 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
        f2 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        f3 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        f4 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        
        T1, T2 = 8, 12
        excitations = np.array([0] * T1)
        orientations = np.array([0] * T2)

        v1 = scipy.signal.convolve2d(subImage, f1, 'same')
        v2 = scipy.signal.convolve2d(subImage, f2, 'same')
        
        v3 = scipy.signal.convolve2d(subImage, f3, 'same')
        v4 = scipy.signal.convolve2d(subImage, f4, 'same')
        
        for x in range(0, height):
            for y in range(0, width):
                if v2[x, y] == 0: v2[x, y] = 1
                if v4[x, y] == 0: v4[x, y] = 1
            
                alpha = math.atan(v1[x, y] / v2[x, y])
                value = math.floor((alpha + (math.pi / 2)) / (math.pi / T1))
                qT1 = self._quantizedT1(T1, alpha)
                excitations[qT1] += value
            
                theta = math.atan(v3[x, y] / v4[x, y])
                mappedTheta = self._mapTheta(v3[x, y], v4[x, y], theta)
                value = math.floor(mappedTheta / ((2 * math.pi) / T2))
                qT2 = self._quantizedT2(T2, mappedTheta)
                orientations[qT2] += value
                
        wld2DHist = np.dot(np.array([excitations]).T, np.array([orientations]))
        descriptor = wld2DHist.flatten()
        
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
