import numpy as np
from multiprocessing import Process, Queue

# Classe do MultiprocessingLBP
class MultiprocessingLBP:
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

    def _process(self, processID, subImage, queue):
        height, width = subImage.shape

        descriptor = [0] * 256
        
        for x in range(1, height - 1):
            previousRow = subImage[x - 1]
            currentRow = subImage[x]
            nextRow = subImage[x + 1]
            for y in range(1, width - 1):
                centerPixel = subImage[x, y]

                # Para otimizar o cálculo do descritor LBP,
                # foi substituido as operações matriciais do módulo "numpy" por 
                # operações lógicas binárias;
                pattern = 0
                pattern = pattern | (1 << 0) if centerPixel < previousRow[y - 1] else pattern
                pattern = pattern | (1 << 1) if centerPixel < previousRow[y] else pattern
                pattern = pattern | (1 << 2) if centerPixel < previousRow[y + 1] else pattern
                pattern = pattern | (1 << 3) if centerPixel < currentRow[y + 1] else pattern
                pattern = pattern | (1 << 4) if centerPixel < nextRow[y + 1] else pattern
                pattern = pattern | (1 << 5) if centerPixel < nextRow[y] else pattern
                pattern = pattern | (1 << 6) if centerPixel < nextRow[y - 1] else pattern
                pattern = pattern | (1 << 7) if centerPixel < currentRow[y - 1] else pattern
                descriptor[pattern] += 1

        descriptor = np.true_divide(descriptor, ((height - 2) * (width - 2)))
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
