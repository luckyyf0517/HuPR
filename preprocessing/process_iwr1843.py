import os
import json
import time
# import cupy as np
import numpy as np
from PIL import Image
from plot_utils import PlotMaps, PlotHeatmaps
from tqdm import tqdm


class RadarObject():
    def __init__(self):
        numGroup = 10
        self.root = 'HuPR'
        self.saveRoot = 'HuPR_official'
        self.sensorType = 'iwr1843'
        self.radarDataFileNameGroup = []
        self.saveDirNameGroup =  []
        self.rgbFileNameGroup = []
        self.jointsFileNameGroup = []
        self.numADCSamples = 256
        self.adcRatio = 4
        self.numAngleBins = self.numADCSamples//self.adcRatio
        self.numEleBins = 8
        self.numRX = 4
        self.numTX = 3
        self.numLanes = 2
        self.framePerSecond = 10
        self.duration = 60
        self.numFrame = self.framePerSecond * self.duration
        self.numChirp = 64 * 3 
        self.idxProcChirp = 64
        self.numGroupChirp = 4
        self.numKeypoints = 14 
        self.xIndices = [-45, -30, -15, 0, 15, 30, 45]
        self.yIndices = [i * 10 for i in range(10)]
        self.initialize(numGroup)

    def initialize(self, numGroup):
        for i in range(1, numGroup + 1):
            radarDataFileName = ['/root/raw_data/hupr/single_' + str(i) + '/hori', 
                                 '/root/raw_data/hupr/single_' + str(i) + '/vert']
            saveDirName = '/root/proc_data/' + self.saveRoot + '/single_' + str(i)
            rgbFileName = 'frames/' + self.root + '/single_' + str(i) + '/processed/images'
            #jointsFileName = '../data/' + self.saveRoot + '/single_' + str(i) + '/annot/hrnet_annot.json'
            self.radarDataFileNameGroup.append(radarDataFileName)
            self.saveDirNameGroup.append(saveDirName)
            self.rgbFileNameGroup.append(rgbFileName)
            #self.jointsFileNameGroup.append(jointsFileName)
    
    def postProcessFFT3D(self, dataFFT):
        dataFFT = np.fft.fftshift(dataFFT, axes=(0, 1))    # [angle, elevation, range]
        dataFFT = np.transpose(dataFFT, (2, 0, 1))  # [range, angle, elevation]
        dataFFT = np.flip(dataFFT, axis=(1, 2))
        return dataFFT

    def getadcDataFromDCA1000(self, fileName):
        adcData = np.fromfile(fileName+'/adc_data.bin', dtype=np.int16)
        fileSize = adcData.shape[0]
        adcData = adcData.reshape(-1, self.numLanes * 2).transpose()
        # for complex data
        fileSize = int(fileSize/2)
        LVDS = np.zeros((2, fileSize))  # seperate each LVDS lane into rows
        LVDS[0, 0::2] = adcData[0]
        LVDS[0, 1::2] = adcData[1]
        LVDS[1, 0::2] = adcData[2]
        LVDS[1, 1::2] = adcData[3]
        # adcData = np.zeros((self.numRX, int(fileSize/self.numRX)), dtype = 'complex_')
        # iter = 0
        # for i in range(0, fileSize, self.numADCSamples * 4):
        #     adcData[0][iter:iter+self.numADCSamples] = LVDS[0][i:i+self.numADCSamples] + np.sqrt(-1+0j)*LVDS[1][i:i+self.numADCSamples]
        #     adcData[1][iter:iter+self.numADCSamples] = LVDS[0][i+self.numADCSamples:i+self.numADCSamples*2] + np.sqrt(-1+0j)*LVDS[1][i+self.numADCSamples:i+self.numADCSamples*2]
        #     adcData[2][iter:iter+self.numADCSamples] = LVDS[0][i+self.numADCSamples*2:i+self.numADCSamples*3] + np.sqrt(-1+0j)*LVDS[1][i+self.numADCSamples*2:i+self.numADCSamples*3]
        #     adcData[3][iter:iter+self.numADCSamples] = LVDS[0][i+self.numADCSamples*3:i+self.numADCSamples*4] + np.sqrt(-1+0j)*LVDS[1][i+self.numADCSamples*3:i+self.numADCSamples*4]
        #     iter = iter + self.numADCSamples
        # adcDataReshape = adcData.reshape(self.numRX, -1, self.numADCSamples)
        adcData = LVDS[0] + 1j * LVDS[1]
        adcDataReshape = adcData.reshape(self.numFrame, self.idxProcChirp, self.numRX * self.numTX, self.numADCSamples)
        print('Shape of radar data:', adcDataReshape.shape)
        return adcDataReshape
    
    def clutterRemoval(self, input_val, axis=0):
        mean_val = np.mean(input_val, axis=axis)
        out_val = input_val - np.expand_dims(mean_val, axis=axis)
        return out_val 

    def generateHeatmap(self, frame):
        # step1: split data
        # dataRadar = np.zeros((self.numRX * 2, self.idxProcChirp, self.numADCSamples), dtype='complex_')
        # dataRadar2 = np.zeros((self.numRX, self.idxProcChirp, self.numADCSamples), dtype='complex_')
        # # Process radar data with TDM-MIMO
        # for idxRX in range(self.numRX):
        #     for idxChirp in range(self.numChirp):
        #         if idxChirp % 3 == 0:
        #             dataRadar[idxRX, idxChirp//3] = frame[idxRX, idxChirp]
        #         if idxChirp % 3 == 1:
        #             dataRadar2[idxRX, idxChirp//3] = frame[idxRX, idxChirp]
        #         elif idxChirp % 3 == 2:
        #             dataRadar[idxRX+4, idxChirp//3] = frame[idxRX, idxChirp]
        dataRadar = frame[:, [0,1,2,3,8,9,10,11], :].transpose(1, 0, 2)
        dataRadar2 = frame[:, [4,5,6,7], :].transpose(1, 0, 2)
        # step1: clutter removal
        dataRadar = self.clutterRemoval(dataRadar, axis=1)
        dataRadar2 = self.clutterRemoval(dataRadar2, axis=1)
        # step2: range-doppler FFT
        # for idxRX in range(self.numRX * 2):
        #     dataRadar[idxRX, :, :] = np.fft.fft2(dataRadar[idxRX, :, :])
        # for idxRX in range(self.numRX * 1):
        #     dataRadar2[idxRX, :, :] = np.fft.fft2(dataRadar2[idxRX, :, :])
        dataRadar = np.fft.fft(dataRadar, axis=1)
        dataRadar = np.fft.fft(dataRadar, axis=2)
        dataRadar2 = np.fft.fft(dataRadar2, axis=1)
        dataRadar2 = np.fft.fft(dataRadar2, axis=2)
        # step3: angle FFT
        padding = ((0, self.numAngleBins - dataRadar.shape[0]), (0,0), (0,0))
        dataRadar = np.pad(dataRadar, padding, mode='constant')
        padding2 = ((2, self.numAngleBins - 4 - 2), (0,0), (0,0))
        dataRadar2 = np.pad(dataRadar2, padding2, mode='constant')
        dataMerge = np.stack((dataRadar, dataRadar2))
        paddingEle = ((0, self.numEleBins - dataMerge.shape[0]), (0,0), (0,0), (0,0))
        dataMerge = np.pad(dataMerge, paddingEle, mode='constant')
        dataMerge[:, 2: 6, :, :] = np.fft.fft(dataMerge[:, 2: 6, :, :], axis=0)
        dataMerge = np.fft.fft(dataMerge, axis=1)
        # for idxChirp in range(self.idxProcChirp):
        #     for idxADC in range(self.numADCSamples):
        #         dataMerge[:, 2, idxChirp, idxADC] = np.fft.fft(dataMerge[:, 2, idxChirp, idxADC])
        #         dataMerge[:, 3, idxChirp, idxADC] = np.fft.fft(dataMerge[:, 3, idxChirp, idxADC])
        #         dataMerge[:, 4, idxChirp, idxADC] = np.fft.fft(dataMerge[:, 4, idxChirp, idxADC])
        #         dataMerge[:, 5, idxChirp, idxADC] = np.fft.fft(dataMerge[:, 5, idxChirp, idxADC])
        #         for idxEle in range(self.numEleBins):
        #             dataMerge[idxEle, :, idxChirp, idxADC] = np.fft.fft(dataMerge[idxEle, :, idxChirp, idxADC])
        # select specific area of ADCSamples (containing signal responses)
        # idxADCSpecific = [i for i in range(94, 30, -1)] # 84, 20
        idxADCSpecific = [i for i in range(31, 95)] # 84, 20
        # select specific velocity information
        chirpPad = self.idxProcChirp//self.numGroupChirp
        dataFFTGroup = np.fft.fftshift(dataMerge, axes=(0, 1, 2))
        dataFFTGroup = dataFFTGroup[:, :, self.idxProcChirp//2 - chirpPad//2: self.idxProcChirp//2 + chirpPad//2, idxADCSpecific].transpose((2, 3, 1, 0))
        dataFFTGroup = np.flip(dataFFTGroup, axis=(1, 2, 3))
        # # shft the velocity information
        # dataTemp = np.zeros((self.idxProcChirp, self.numADCSamples//self.adcRatio, self.numAngleBins, self.numEleBins), dtype='complex_')
        # dataFFTGroup = np.zeros((self.idxProcChirp//self.numGroupChirp, self.numADCSamples//self.adcRatio, self.numAngleBins, self.numEleBins), dtype='complex_')
        # for idxEle in range(self.numEleBins):
        #     for idxRX in range(self.numAngleBins):
        #         for idxADC in range(self.numADCSamples//self.adcRatio):
        #             dataTemp[:, idxADC, idxRX, idxEle] = dataMerge[idxEle, idxRX, :, idxADCSpecific[idxADC]]
        #             dataTemp[:, idxADC, idxRX, idxEle] = np.fft.fftshift(dataTemp[:, idxADC, idxRX, idxEle], axes=(0))
        # i = 0
        # for idxChirp in range(self.idxProcChirp//2 - chirpPad//2, self.idxProcChirp//2 + chirpPad//2):
        #     dataFFTGroup[i, :, :, :] = self.postProcessFFT3D(np.transpose(dataTemp[idxChirp, :, :, :], (1, 2, 0)))
        #     i += 1
        return dataFFTGroup  
    
    def saveDataAsFigure(self, img, joints, output, visDirName, idxFrame, output2=None):
        heatmap = PlotHeatmaps(joints, self.numKeypoints)
        PlotMaps(visDirName, self.xIndices, self.yIndices, 
        idxFrame, output, img, heatmap, output2)
    
    def saveRadarData(self, matrix, dirName, idxFrame):
        if not os.path.exists(dirName): 
            os.makedirs(dirName)
        dirSave = dirName + ('/%09d' % idxFrame) + '.npy'
        np.save(dirSave, matrix)

    def processRadarDataHoriVert(self):
        #numpoints = []
        for idxName in range(len(self.radarDataFileNameGroup)):
            print('Processing', self.radarDataFileNameGroup[idxName][0])
            adcDataHori = self.getadcDataFromDCA1000(self.radarDataFileNameGroup[idxName][0])
            adcDataVert = self.getadcDataFromDCA1000(self.radarDataFileNameGroup[idxName][1])
            for idxFrame in tqdm(range(0, self.numFrame)):
                frameHori = adcDataHori[idxFrame]
                frameVert = adcDataVert[idxFrame]
                tic = time.time()
                outputHori = self.generateHeatmap(frameHori)
                outputVert = self.generateHeatmap(frameVert)
                self.saveRadarData(outputHori, self.saveDirNameGroup[idxName] + '/hori', idxFrame)
                self.saveRadarData(outputVert, self.saveDirNameGroup[idxName] + '/vert', idxFrame)
                # print('%s, finished frame %d' % (self.radarDataFileNameGroup[idxName][0], idxFrame), end='\r')
    
    def loadDataPlot(self):
        for idxName in range(len(self.radarDataFileNameGroup)):
            with open(self.jointsFileNameGroup[idxName], "r") as fp:
                annotGroup = json.load(fp)
            for idxFrame in range(0,self.numFrame):
                hori_path = self.saveDirNameGroup[idxName] + '/hori' + ('/%09d' % idxFrame) + '.npy'
                vert_path = self.saveDirNameGroup[idxName] + '/vert' + ('/%09d' % idxFrame) + '.npy'
                outputHori = np.load(hori_path)
                outputVert = np.load(vert_path)
                outputHori = np.mean(np.abs(outputHori), axis=(0, 3))
                outputVert = np.mean(np.abs(outputVert), axis=(0, 3))
                visDirName = self.saveDirNameGroup[idxName] + '/visualization' + ('/%09d.png' % idxFrame)
                img = np.array(Image.open(self.rgbFileNameGroup[idxName] + "/%09d.jpg" % idxFrame).convert('RGB'))
                joints = annotGroup[idxFrame]['joints']
                self.saveDataAsFigure(img, joints, outputHori, visDirName, idxFrame, outputVert)
                print('%s, finished frame %d' % (self.radarDataFileNameGroup[idxName][0], idxFrame), end='\r')

if __name__ == "__main__":
    visualization = False
    radarObject = RadarObject()
    if not visualization:
        radarObject.processRadarDataHoriVert()
    #else:
    #    radarObject.loadDataPlot()
