# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 00:18:47 2016

@author: Sergey
"""

from astropy.io import fits
import numpy as NP
from astropy import coordinates
from astropy import constants
from BadaryRAO import BadaryRAO
from scipy.optimize import least_squares, basinhopping
import sunpy.coordinates
import base2uvw_36
from skimage.transform import warp, AffineTransform
import scipy.signal
import time
from ZirinTb import ZirinTb
import json
import skimage.measure
from pathlib import Path

class SrhFitsFile():
    def __init__(self, name, sizeOfUv, flux_norm = True):
        self.fitsFileName = name
        self.omegaEarth = coordinates.earth.OMEGA_EARTH.to_value()
        self.isOpen = False;
        self.calibIndex = 0;
        self.frequencyChannel = 0;
        self.centerP = 0.;
        self.deltaP = 0.005;
        self.centerQ = 0.;
        self.deltaQ = 0.005;
        self.centerH = 0.;
        self.deltaH = 4.9/3600.*NP.pi;
        self.centerD = 0.;
        self.deltaD = 4.9/3600.*NP.pi;
        self.antNumberEW = 97
        self.antNumberN = 31
        self.averageCalib = False
        self.useNonlinearApproach = True
        self.obsObject = 'Sun'
        self.fringeStopping = False

        self.centering_ftol = 1e-3

        self.badAntsLcp = NP.zeros(128)
        self.badAntsRcp = NP.zeros(128)
        self.sizeOfUv = sizeOfUv
        self.baselines = 5
        self.flagsIndexes = []
        self.arcsecPerPixel = 4.91104/2
        self.ZirinQSunTb = ZirinTb()
        self.convolutionNormCoef = 44.8
        self.useRLDif = False
        self.flux_calibrated = False
        self.corr_amp_exist = False
        
        self.open(name, flux_norm = flux_norm)
                         
    def open(self,fitsNames, flux_norm):
        if type(fitsNames) is not list: fitsNames = [fitsNames]
        if fitsNames[0]:
            try:
                self.hduList = fits.open(fitsNames[0])
                self.isOpen = True
                self.dateObs = self.hduList[0].header['DATE-OBS'] + 'T' + self.hduList[0].header['TIME-OBS']
                self.antennaNumbers = self.hduList[2].data['ant_index']
                self.antennaNumbers = NP.reshape(self.antennaNumbers,self.antennaNumbers.size)
                self.antennaNames = self.hduList[2].data['ant_name']
                self.antennaNames = NP.reshape(self.antennaNames,self.antennaNames.size)
                self.antennaA = self.hduList[4].data['ant_A']
                self.antennaA = NP.reshape(self.antennaA,self.antennaA.size)
                self.antennaB = self.hduList[4].data['ant_B']
                self.antennaB = NP.reshape(self.antennaB,self.antennaB.size)
                self.antX = self.hduList[3].data['ant_X']
                self.antY = self.hduList[3].data['ant_Y']
                self.uvLcp = NP.zeros((self.sizeOfUv,self.sizeOfUv),dtype=complex)
                self.uvRcp = NP.zeros((self.sizeOfUv,self.sizeOfUv),dtype=complex)
                self.freqList = self.hduList[1].data['frequency'];
                self.freqListLength = self.freqList.size;
                self.dataLength = self.hduList[1].data['time'].size // self.freqListLength;
                self.freqTime = self.hduList[1].data['time']
                self.validScansLcp = NP.ones((self.freqListLength,self.dataLength), dtype = bool)
                self.validScansRcp = NP.ones((self.freqListLength,self.dataLength), dtype = bool)
                try:
                    self.freqTimeLcp = self.hduList[1].data['time_lcp']
                    self.freqTimeRcp = self.hduList[1].data['time_rcp']
                except:
                    pass
                self.visListLength = self.hduList[1].data['vis_lcp'].size // self.freqListLength // self.dataLength;
                self.visLcp = NP.reshape(self.hduList[1].data['vis_lcp'],(self.freqListLength,self.dataLength,self.visListLength));
                self.visRcp = NP.reshape(self.hduList[1].data['vis_rcp'],(self.freqListLength,self.dataLength,self.visListLength));
                # self.visLcp /= float(self.hduList[0].header['VIS_MAX'])
                # self.visRcp /= float(self.hduList[0].header['VIS_MAX'])
                self.ampLcp = NP.reshape(self.hduList[1].data['amp_lcp'],(self.freqListLength,self.dataLength,self.antennaNumbers.size));
                self.ampRcp = NP.reshape(self.hduList[1].data['amp_rcp'],(self.freqListLength,self.dataLength,self.antennaNumbers.size));
                ampScale = float(self.hduList[0].header['VIS_MAX']) / 128.
                self.ampLcp = self.ampLcp.astype(float) / ampScale
                self.ampRcp = self.ampRcp.astype(float) / ampScale
                try:
                    self.correctSubpacketsNumber = int(self.hduList[0].header['SUBPACKS'])
                    self.subpacketLcp = self.hduList[1].data['spacket_lcp']
                    self.subpacketRcp = self.hduList[1].data['spacket_rcp']
                    self.validScansLcp = self.subpacketLcp==self.correctSubpacketsNumber
                    self.validScansRcp = self.subpacketRcp==self.correctSubpacketsNumber
                    self.visLcp[~self.validScansLcp] = 0
                    self.visRcp[~self.validScansRcp] = 0
                    self.ampLcp[~self.validScansLcp] = 1
                    self.ampRcp[~self.validScansRcp] = 1
                except:
                    pass
                try:
                    self.ampLcp_c = NP.reshape(self.hduList[1].data['amp_c_lcp'],(self.freqListLength,self.dataLength,self.antennaNumbers.size));
                    self.ampRcp_c = NP.reshape(self.hduList[1].data['amp_c_rcp'],(self.freqListLength,self.dataLength,self.antennaNumbers.size));
                    # self.ampLcp_c = self.ampLcp_c.astype(float) / ampScale
                    # self.ampRcp_c = self.ampRcp_c.astype(float) / ampScale
                    self.corr_amp_exist = True
                    self.ampLcp_c[self.ampLcp_c <= 0.01] = 1e6
                    self.ampRcp_c[self.ampRcp_c <= 0.01] = 1e6
                    antA_new = self.antennaA.copy()
                    antA_new[self.antennaA<98] += 30
                    antA_new[self.antennaA>=98] -= 98
                    antB_new = self.antennaB.copy()
                    antB_new[self.antennaB<98] += 30
                    antB_new[self.antennaB>=98] -= 98
                    self.visLcp = self.visLcp / ((NP.sqrt(self.ampLcp_c[:,:,antA_new] * self.ampLcp_c[:,:,antB_new])))
                    self.visRcp = self.visRcp / ((NP.sqrt(self.ampRcp_c[:,:,antA_new] * self.ampRcp_c[:,:,antB_new])))
                except:
                    pass
                self.antZeroRow = self.hduList[3].data['ant_zero_row'][:97]
                self.RAO = BadaryRAO(self.dateObs.split('T')[0], 9.8, observedObject = self.obsObject)
                # try:
                #     client = Client('http://ephemeris.rao.istp.ac.ru/?wsdl')
                #     result = client.service.Ephemeride('SSRT','sun',self.dateObs)
                #     self.pAngle = NP.deg2rad(float(result[0]['PAngle']))
                # except:
                self.pAngle = NP.deg2rad(sunpy.coordinates.sun.P(self.dateObs).to_value())
                self.getHourAngle(0)
                
                self.ewAntPhaLcp = NP.zeros((self.freqListLength, self.antNumberEW))
                self.nAntPhaLcp = NP.zeros((self.freqListLength, self.antNumberN))
                self.ewAntPhaRcp = NP.zeros((self.freqListLength, self.antNumberEW))
                self.nAntPhaRcp = NP.zeros((self.freqListLength, self.antNumberN))
                self.ewLcpPhaseCorrection = NP.zeros((self.freqListLength, self.antNumberEW))
                self.ewRcpPhaseCorrection = NP.zeros((self.freqListLength, self.antNumberEW))
                self.nLcpPhaseCorrection = NP.zeros((self.freqListLength, self.antNumberN))
                self.nRcpPhaseCorrection = NP.zeros((self.freqListLength, self.antNumberN))
                self.sLcpStair = NP.zeros(self.freqListLength)
                self.sRcpStair = NP.zeros(self.freqListLength)
                self.nSolarPhase = NP.zeros(self.freqListLength)
                self.ewSolarPhase = NP.zeros(self.freqListLength)
                
                self.ewAntAmpLcp = NP.ones((self.freqListLength, self.antNumberEW))
                self.nAntAmpLcp = NP.ones((self.freqListLength, self.antNumberN))
                self.ewAntAmpRcp = NP.ones((self.freqListLength, self.antNumberEW))
                self.nAntAmpRcp = NP.ones((self.freqListLength, self.antNumberN))
                
                self.ewPhaseDif = NP.zeros_like(self.ewAntPhaLcp)
                self.nPhaseDif = NP.zeros_like(self.nAntPhaLcp)
                self.ewAmpDif = NP.zeros_like(self.ewAntAmpLcp)
                self.nAmpDif = NP.zeros_like(self.nAntAmpLcp)
                
                self.nLcpStair = NP.zeros(self.freqListLength)
                self.nRcpStair = NP.zeros(self.freqListLength)
                self.ewSlopeLcp = NP.zeros(self.freqListLength)
                self.nSlopeLcp = NP.zeros(self.freqListLength)
                self.ewSlopeRcp = NP.zeros(self.freqListLength)
                self.nSlopeRcp = NP.zeros(self.freqListLength)
                self.diskLevelLcp = NP.ones(self.freqListLength)
                self.diskLevelRcp = NP.ones(self.freqListLength)
                self.lm_hd_relation = NP.ones(self.freqListLength)
                
                self.flags_ew = NP.array((), dtype = int)
                self.flags_n = NP.array((), dtype = int)
                
                self.fluxLcp = NP.zeros(self.freqListLength)
                self.fluxRcp = NP.zeros(self.freqListLength)
                
                x_size = (self.baselines-1)*2 + self.antNumberEW + self.antNumberN
                self.x_ini_lcp = NP.full((self.freqListLength, x_size*2+1), NP.concatenate((NP.ones(x_size+1), NP.zeros(x_size))))
                self.x_ini_rcp = NP.full((self.freqListLength, x_size*2+1), NP.concatenate((NP.ones(x_size+1), NP.zeros(x_size))))
                self.calibrationResultLcp = NP.zeros_like(self.x_ini_lcp)
                self.calibrationResultRcp = NP.zeros_like(self.x_ini_rcp)
                
                self.lcpShift = NP.ones(self.freqListLength) # 0-frequency component in the spectrum
                self.rcpShift = NP.ones(self.freqListLength)
                
                self.beam_sr = NP.ones(self.freqListLength)
                
            except FileNotFoundError:
                print('File %s  not found'%fitsNames[0]);
             
        if len(fitsNames) > 1:
            for fitsName in fitsNames[1:]:
                self.append(fitsName)
        if flux_norm and self.corr_amp_exist:
            self.normalizeFlux()
            self.beam()
    
    def append(self,name):
        try:
            hduList = fits.open(name);
            freqTime = hduList[1].data['time']
            dataLength = hduList[1].data['time'].size // self.freqListLength;
            visLcp = NP.reshape(hduList[1].data['vis_lcp'],(self.freqListLength,dataLength,self.visListLength));
            visRcp = NP.reshape(hduList[1].data['vis_rcp'],(self.freqListLength,dataLength,self.visListLength));
            # visLcp /= float(self.hduList[0].header['VIS_MAX'])
            # visRcp /= float(self.hduList[0].header['VIS_MAX'])
            ampLcp = NP.reshape(hduList[1].data['amp_lcp'],(self.freqListLength,dataLength,self.antennaNumbers.size));
            ampRcp = NP.reshape(hduList[1].data['amp_rcp'],(self.freqListLength,dataLength,self.antennaNumbers.size));
            ampScale = float(self.hduList[0].header['VIS_MAX']) / 128.
            ampLcp = ampLcp.astype(float) / ampScale
            ampRcp = ampRcp.astype(float) / ampScale
            validScansLcp = NP.ones((self.freqListLength, dataLength), dtype = bool)
            validScansRcp = NP.ones((self.freqListLength, dataLength), dtype = bool)
            try:
                subpacketLcp = self.hduList[1].data['spacket_lcp']
                subpacketRcp = self.hduList[1].data['spacket_rcp']
                self.subpacketLcp = NP.concatenate((self.subpacketLcp, subpacketLcp), axis = 1)
                self.subpacketRcp = NP.concatenate((self.subpacketRcp, subpacketRcp), axis = 1)
                validScansLcp = subpacketLcp==self.correctSubpacketsNumber
                validScansRcp = subpacketRcp==self.correctSubpacketsNumber
                visLcp[~validScansLcp] = 0
                visRcp[~validScansRcp] = 0
                ampLcp[~validScansLcp] = 1
                ampRcp[~validScansRcp] = 1
            except:
                pass
            try:
                ampLcp_c = NP.reshape(hduList[1].data['amp_c_lcp'],(self.freqListLength,dataLength,self.antennaNumbers.size));
                ampRcp_c = NP.reshape(hduList[1].data['amp_c_rcp'],(self.freqListLength,dataLength,self.antennaNumbers.size));
                # ampLcp_c = ampLcp_c.astype(float) / ampScale
                # ampRcp_c = ampRcp_c.astype(float) / ampScale
                ampLcp_c[ampLcp_c <= 0.01] = 1e6
                ampRcp_c[ampRcp_c <= 0.01] = 1e6
                antA_new = self.antennaA.copy()
                antA_new[self.antennaA<98] += 30
                antA_new[self.antennaA>=98] -= 98
                antB_new = self.antennaB.copy()
                antB_new[self.antennaB<98] += 30
                antB_new[self.antennaB>=98] -= 98
                visLcp = visLcp/NP.sqrt(NP.abs(ampLcp_c[:,:,antA_new]) * NP.abs(ampLcp_c[:,:,antB_new]))
                visRcp = visRcp/NP.sqrt(NP.abs(ampRcp_c[:,:,antA_new]) * NP.abs(ampRcp_c[:,:,antB_new]))
                self.ampLcp_c = NP.concatenate((self.ampLcp_c, ampLcp_c), axis = 1)
                self.ampRcp_c = NP.concatenate((self.ampRcp_c, ampRcp_c), axis = 1)
            except:
                pass
            try:
                freqTimeLcp = hduList[1].data['time_lcp']
                freqTimeRcp = hduList[1].data['time_rcp']
                self.freqTimeLcp = NP.concatenate((self.freqTimeLcp, freqTimeLcp), axis = 1)
                self.freqTimeRcp = NP.concatenate((self.freqTimeRcp, freqTimeRcp), axis = 1)
            except:
                pass
            
            self.freqTime = NP.concatenate((self.freqTime, freqTime), axis = 1)
            self.visLcp = NP.concatenate((self.visLcp, visLcp), axis = 1)
            self.visRcp = NP.concatenate((self.visRcp, visRcp), axis = 1)
            self.ampLcp = NP.concatenate((self.ampLcp, ampLcp), axis = 1)
            self.ampRcp = NP.concatenate((self.ampRcp, ampRcp), axis = 1)
            self.validScansLcp = NP.concatenate((self.validScansLcp, validScansLcp), axis = 1)
            self.validScansRcp = NP.concatenate((self.validScansRcp, validScansRcp), axis = 1)
            self.dataLength += dataLength
            hduList.close()

        except FileNotFoundError:
            print('File %s  not found'%name);
            
    def normalizeFlux(self):
        file = Path(__file__).resolve()
        parent = str(file.parent)
        zerosFits = fits.open(parent + '/srh_0306_cp_zeros.fits')
        corrZeros = zerosFits[2].data['corrI']
        fluxZeros = zerosFits[2].data['fluxI']

        fluxNormFits = fits.open(parent + '/srh_0306_cp_fluxNorm.fits')
        fluxNormI = fluxNormFits[2].data['fluxNormI']
        
        # antA_new = self.antennaA.copy()
        # antA_new[self.antennaA<98] += 30
        # antA_new[self.antennaA>=98] -= 98
        
        # antB_new = self.antennaB.copy()
        # antB_new[self.antennaB<98] += 30
        # antB_new[self.antennaB>=98] -= 98
        
        # self.antAmp_Lcp = NP.sqrt(NP.abs(self.ampLcp_c[:,:,antA_new]) * NP.abs(self.ampLcp_c[:,:,antB_new]))
        # self.antAmp_Rcp = NP.sqrt(NP.abs(self.ampRcp_c[:,:,antA_new]) * NP.abs(self.ampRcp_c[:,:,antB_new]))
        
        # self.visLcp = self.visLcp/self.antAmp_Lcp
        # self.visRcp = self.visRcp/self.antAmp_Rcp
        
        ampFluxRcp = NP.mean(self.ampRcp, axis = 2)
        ampFluxLcp = NP.mean(self.ampLcp, axis = 2)
        
        for ff in range(self.freqListLength):
            # self.setFrequencyChannel(ff)
            
            ampFluxRcp[ff,:] -= fluxZeros[ff]
            ampFluxRcp[ff,:] *= fluxNormI[ff] 
            ampFluxLcp[ff,:] -= fluxZeros[ff]
            ampFluxLcp[ff,:] *= fluxNormI[ff] 
            
            # lam = scipy.constants.c/(self.freqList[ff]*1e3)
            
            # self.tempLcp[ff] = NP.mean(ampFluxLcp[ff]) * lam**2 / (2*scipy.constants.k * self.beam_sr[ff])
            # self.tempRcp[ff] = NP.mean(ampFluxRcp[ff]) * lam**2 / (2*scipy.constants.k * self.beam_sr[ff])
            
            self.fluxLcp[ff] = NP.mean(ampFluxLcp[ff])
            self.fluxRcp[ff] = NP.mean(ampFluxRcp[ff])
            
            self.visLcp[ff,:,:] *= NP.mean(self.fluxLcp[ff])
            self.visRcp[ff,:,:] *= NP.mean(self.fluxRcp[ff])
            
            self.visLcp[ff,:,:] *= 2 # flux is divided by 2 for R and L
            self.visRcp[ff,:,:] *= 2
            
        self.flux_calibrated = True
            
    def beam(self):
        self.setFrequencyChannel(0)
        self.vis2uv(0, PSF = True, average=20)
        self.uv2lmImage()
        self.lm2Heliocentric(image_scale = 2)
        arcsecPerPix = self.arcsecPerPixel / 2.
        beam = self.lcp
        contours = (skimage.measure.find_contours(beam, 0.5*beam.max()))[0]
        con = NP.zeros_like(contours)
        con[:,1] = contours[:,0]
        con[:,0] = contours[:,1]
        sunEll = skimage.measure.EllipseModel()
        sunEll.estimate(con)
        major = NP.deg2rad(sunEll.params[2] * arcsecPerPix / 3600.)
        minor = NP.deg2rad(sunEll.params[3] * arcsecPerPix / 3600.)
        self.beam_sr[0] = NP.pi * major * minor / NP.log(2)
        for ff in range(1, self.freqListLength):
            self.beam_sr[ff] = self.beam_sr[0] * (self.freqList[0]/self.freqList[ff])**2
        """
        In original formula (https://science.nrao.edu/facilities/vla/proposing/TBconv)
        theta_maj ang theta_min are full widths of an ellipse, that is why there is
        4 in denominator.
        Here major and minor are semi-axes.
        """
        
            
    def calibrate(self, freq = 'all', phaseCorrect = True, amplitudeCorrect = True, average = 0):
        if freq == 'all':
            self.calculatePhaseCalibration()
            for freq in range(self.freqListLength):
                self.setFrequencyChannel(freq)
                self.vis2uv(scan = self.calibIndex, phaseCorrect=phaseCorrect, amplitudeCorrect=amplitudeCorrect, average=average)
                self.centerDisk()
        else:
            self.setFrequencyChannel(freq)
            self.solarPhase(freq)
            self.updateAntennaPhase(freq)
            self.vis2uv(scan = self.calibIndex, phaseCorrect=phaseCorrect, amplitudeCorrect=amplitudeCorrect, average=average)
            self.centerDisk()
            
    def image(self, freq, scan, average = 0, polarization = 'both', phaseCorrect = True, amplitudeCorrect = True, frame = 'heliocentric'):
        self.setFrequencyChannel(freq)    
        self.vis2uv(scan = scan, phaseCorrect=phaseCorrect, amplitudeCorrect=amplitudeCorrect, average=average)
        self.uv2lmImage()
        if frame == 'heliocentric':
            self.lm2Heliocentric()
        if polarization == 'both':
            return NP.flip(self.lcp, 0), NP.flip(self.rcp, 0)
        elif polarization == 'lcp':
            return NP.flip(self.lcp, 0)
        elif polarization == 'rcp':
            return NP.flip(self.rcp, 0)
        
    def saveGains(self, filename):
        currentGainsDict = {}
        currentGainsDict['ewPhaseLcp'] = (self.ewAntPhaLcp + self.ewLcpPhaseCorrection).tolist()
        currentGainsDict['nPhaseLcp'] = (self.nAntPhaLcp + self.nLcpPhaseCorrection).tolist()
        currentGainsDict['ewPhaseRcp'] = (self.ewAntPhaRcp + self.ewRcpPhaseCorrection).tolist()
        currentGainsDict['nPhaseRcp'] = (self.nAntPhaRcp + self.nRcpPhaseCorrection).tolist()
        currentGainsDict['ewAmpLcp'] = self.ewAntAmpLcp.tolist()
        currentGainsDict['nAmpLcp'] = self.nAntAmpLcp.tolist()
        currentGainsDict['ewAmpRcp'] = self.ewAntAmpRcp.tolist()
        currentGainsDict['nAmpRcp'] = self.nAntAmpRcp.tolist()
        currentGainsDict['rcpShift'] = self.rcpShift.tolist()
        currentGainsDict['lcpShift'] = self.lcpShift.tolist()
        currentGainsDict['lm_hd_relation'] = self.lm_hd_relation.tolist()
        with open(filename, 'w') as saveGainFile:
            json.dump(currentGainsDict, saveGainFile)
            
    def loadGains(self, filename):
        with open(filename,'r') as readGainFile:
            currentGains = json.load(readGainFile)
        self.ewAntPhaLcp = NP.array(currentGains['ewPhaseLcp'])
        self.ewAntPhaRcp = NP.array(currentGains['ewPhaseRcp'])
        self.ewAntAmpLcp = NP.array(currentGains['ewAmpLcp'])
        self.ewAntAmpRcp = NP.array(currentGains['ewAmpRcp'])
        
        try:
            self.nAntPhaLcp = NP.array(currentGains['nPhaseLcp'])
            self.nAntPhaRcp = NP.array(currentGains['nPhaseRcp'])
            self.nAntAmpLcp = NP.array(currentGains['nAmpLcp'])
            self.nAntAmpRcp = NP.array(currentGains['nAmpRcp'])
        except:
            pass
        try:
            self.nAntPhaLcp = NP.array(currentGains['nsPhaseLcp'])
            self.nAntPhaRcp = NP.array(currentGains['nsPhaseRcp'])
            self.nAntAmpLcp = NP.array(currentGains['nsAmpLcp'])
            self.nAntAmpRcp = NP.array(currentGains['nsAmpRcp'])
        except:
            pass
        
        self.rcpShift = NP.array(currentGains['rcpShift'])
        self.lcpShift = NP.array(currentGains['lcpShift'])
        self.lm_hd_relation = NP.array(currentGains['lm_hd_relation'])
       
    def saveRLdif(self):
        filename = self.fitsFileName[:-3]+'json'
        self.ewPhaseDif = self.ewAntPhaLcp + self.ewLcpPhaseCorrection - (self.ewAntPhaRcp + self.ewRcpPhaseCorrection)
        self.nPhaseDif = self.nAntPhaLcp + self.nLcpPhaseCorrection - (self.nAntPhaRcp + self.nRcpPhaseCorrection)
        self.ewAmpDif = self.ewAntAmpLcp/self.ewAntAmpRcp
        self.nAmpDif = self.nAntAmpLcp/self.nAntAmpRcp
        RLDif_dict = {}
        RLDif_dict['ewPhaseDif'] = self.ewPhaseDif.tolist()
        RLDif_dict['nPhaseDif'] = self.nPhaseDif.tolist()
        RLDif_dict['ewAmpDif'] = self.ewAmpDif.tolist()
        RLDif_dict['nAmpDif'] = self.nAmpDif.tolist()
        with open(filename, 'w') as saveRLDifFile:
            json.dump(RLDif_dict, saveRLDifFile)
       
    def loadRLdif(self, filename):
        with open(filename,'r') as readRLDifFile:
            RLDif = json.load(readRLDifFile)
        self.ewPhaseDif = NP.array(RLDif['ewPhaseDif'])
        self.nPhaseDif = NP.array(RLDif['nPhaseDif'])
        self.ewAmpDif = NP.array(RLDif['ewAmpDif'])
        self.nAmpDif = NP.array(RLDif['nAmpDif'])

    def changeObject(self, obj):
        self.obsObject = obj
        self.RAO = BadaryRAO(self.dateObs.split('T')[0], observedObject = self.obsObject)
        
    def getHourAngle(self, scan):
        self.hAngle = self.omegaEarth * (self.freqTime[self.frequencyChannel, scan] - self.RAO.culmination)
        if self.hAngle > 1.5*NP.pi:
            self.hAngle -= 2*NP.pi
        return self.hAngle
        
    def setSizeOfUv(self, sizeOfUv):
        self.sizeOfUv = sizeOfUv
        self.uvLcp = NP.zeros((self.sizeOfUv,self.sizeOfUv),dtype=complex);
        self.uvRcp = NP.zeros((self.sizeOfUv,self.sizeOfUv),dtype=complex);
        
    def getDeclination(self):
        return self.RAO.declination

    def getPQScale(self, size, FOV):
        self.cosP = NP.sin(self.hAngle) * NP.cos(self.RAO.declination)
        self.cosQ = NP.cos(self.hAngle) * NP.cos(self.RAO.declination) * NP.sin(self.RAO.observatory.lat) - NP.sin(self.RAO.declination) * NP.cos(self.RAO.observatory.lat)
        FOV_p = 2.*(constants.c / (self.freqList[self.frequencyChannel]*1e3)) / (self.RAO.base*NP.sqrt(1. - self.cosP**2.));
        FOV_q = 2.*(constants.c / (self.freqList[self.frequencyChannel]*1e3)) / (self.RAO.base*NP.sqrt(1. - self.cosQ**2.));
        
        return [int(size*FOV/FOV_p.to_value()), int(size*FOV/FOV_q.to_value())]
        
    def getPQ2HDMatrix(self):
        gP =  NP.arctan(NP.tan(self.hAngle)*NP.sin(self.RAO.declination));
        gQ =  NP.arctan(-(NP.sin(self.RAO.declination) / NP.tan(self.hAngle) + NP.cos(self.RAO.declination) / (NP.sin(self.hAngle)*NP.tan(self.RAO.observatory.lat))));
        
        if self.hAngle > 0:
            gQ = NP.pi + gQ;
        g = gP - gQ;
          
        pqMatrix = NP.zeros((3,3))
        pqMatrix[0, 0] =  NP.cos(gP) - NP.cos(g)*NP.cos(gQ)
        pqMatrix[0, 1] = -NP.cos(g)*NP.cos(gP) + NP.cos(gQ)
        pqMatrix[1, 0] =  NP.sin(gP) - NP.cos(g)*NP.sin(gQ)
        pqMatrix[1, 1] = -NP.cos(g)*NP.sin(gP) + NP.sin(gQ)
        pqMatrix /= NP.sin(g)**2.
        pqMatrix[2, 2] = 1.
        return pqMatrix
        
    def close(self):
        self.hduList.close();
        
    def flag(self, names):
        nameList = names.split(',')
        for i in range(len(nameList)):
            ind = NP.where(self.antennaNames == nameList[i])[0]
            if len(ind):
                self.flagsIndexes.append(int(self.antennaNumbers[ind[0]]))
        self.flagVis = NP.array([], dtype = int)
        for i in range(len(self.flagsIndexes)):
            ind = NP.where(self.antennaA == self.flagsIndexes[i])[0]
            if len(ind):
                self.flagVis = NP.append(self.flagVis, ind)
            ind = NP.where(self.antennaB == self.flagsIndexes[i])[0]
            if len(ind):
                self.flagVis = NP.append(self.flagVis, ind)
        self.visLcp[:,:,self.flagVis] = 0
        self.visRcp[:,:,self.flagVis] = 0
        
    def visIndex2antIndex(self, visIndex):
        if visIndex > self.antennaA.size or visIndex < 0:
            Exception('visIndex is out of range')
        else:
            return self.antennaA[visIndex], self.antennaB[visIndex]
    
    def antIndex2visIndex(self, antA, antB):
        if antA > self.antennaA.size or antA < 0:
            Exception('antA index is out of range')
        elif antB > self.antennaB.size or antB < 0:
            Exception('antB index is out of range')
        else:
            try:
                return NP.where((self.antennaA==antA) & (self.antennaB==antB))[0][0]
            except:
                return NP.where((self.antennaA==antB) & (self.antennaB==antA))[0][0]
                
    def setBaselinesNumber(self, value):
        self.baselines = value
        x_size = (self.baselines-1)*2 + self.antNumberEW + self.antNumberN
        self.x_ini_lcp = NP.full((self.freqListLength, x_size*2+1), NP.concatenate((NP.ones(x_size+1), NP.zeros(x_size))))
        self.x_ini_rcp = NP.full((self.freqListLength, x_size*2+1), NP.concatenate((NP.ones(x_size+1), NP.zeros(x_size))))
        self.calibrationResultLcp = NP.zeros_like(self.x_ini_lcp)
        self.calibrationResultRcp = NP.zeros_like(self.x_ini_rcp)

    def phaMatrixGenPairsEWN(self, pairs, antNumberEW, antNumberN):
        rowsEW = int(((antNumberEW - 1) + (antNumberEW - pairs))/2 * pairs)
        rowsN = int(((antNumberN) + (antNumberN + 1 - pairs))/2 * pairs)
        colsEW = antNumberEW + pairs
        colsN = antNumberN + pairs
        phaMatrix = NP.zeros((rowsEW + rowsN, colsEW + colsN))
        for pair in range(pairs):
            row0 = int(((antNumberEW - 1) + (antNumberEW - pair))/2 * pair)
            row1 = row0 + (antNumberEW - pair - 1)
            phaMatrix[row0:row1,pair] = 1
            for phaPair in range(antNumberEW - pair - 1):
                phaMatrix[phaPair + row0, phaPair + 2*pairs] = 1
                phaMatrix[phaPair + row0, phaPair + 2*pairs + (pair + 1)] = -1
            row0 = int(((antNumberN) + (antNumberN + 1 - pair))/2 * pair)
            row1 = row0 + (antNumberN - pair)
            phaMatrix[row0 + rowsEW:row1 + rowsEW,pairs + pair] = 1
            for phaPair in range(antNumberN - pair):
                if phaPair == 0:
                    phaMatrix[rowsEW + row0, 2*pairs + 32] = 1
                else:
                    phaMatrix[phaPair + rowsEW + row0, 2*pairs + phaPair + antNumberEW - 1] = 1
                phaMatrix[phaPair + rowsEW + row0, 2*pairs + phaPair + antNumberEW - 1 + (pair + 1)] = -1
        return phaMatrix.copy()
    
    def calculatePhaseCalibration(self, baselinesNumber = 5, lcp = True, rcp = True):
       for freq in range(self.freqListLength):
           self.solarPhase(freq)
           self.updateAntennaPhase(freq, baselinesNumber, lcp = lcp, rcp = rcp)
           
    def calculateAmpCalibration(self, baselinesNumber = 5):
       for freq in range(self.freqListLength):
           self.calculateAmplitude_linear(freq, baselinesNumber)

    def updateAntennaPhase(self, freqChannel, baselinesNumber = 5, lcp = True, rcp = True):
        if self.useNonlinearApproach:
            if lcp:
                self.calculatePhaseLcp_nonlinear_new(freqChannel, baselinesNumber = baselinesNumber)
            if rcp:
                self.calculatePhaseRcp_nonlinear_new(freqChannel, baselinesNumber = baselinesNumber)
            if rcp and lcp:
                flags_ew_lcp = NP.where(self.ewAntAmpLcp[freqChannel] == 1e6)[0]
                flags_ew_rcp = NP.where(self.ewAntAmpRcp[freqChannel] == 1e6)[0]
                self.flags_ew = NP.unique(NP.append(flags_ew_lcp, flags_ew_rcp))
                flags_n_lcp = NP.where(self.nAntAmpLcp[freqChannel] == 1e6)[0]
                flags_n_rcp = NP.where(self.nAntAmpRcp[freqChannel] == 1e6)[0]
                self.flags_n = NP.unique(NP.append(flags_n_lcp, flags_n_rcp))
                self.ewAntAmpLcp[freqChannel][self.flags_ew] = 1e6
                self.nAntAmpLcp[freqChannel][self.flags_n] = 1e6
                self.ewAntAmpRcp[freqChannel][self.flags_ew] = 1e6
                self.nAntAmpRcp[freqChannel][self.flags_n] = 1e6
        else:
            self.calculatePhase_linear(freqChannel, baselinesNumber = baselinesNumber)
            
    def solarPhase(self, freq):
        u,v,w = base2uvw_36.base2uvw(self.hAngle, self.RAO.declination, 98, 99)
        baseWave = NP.sqrt(u**2+v**2)*self.freqList[freq]*1e3/constants.c.to_value()
        if baseWave > 120:
            self.nSolarPhase[freq] = NP.pi
        else:
            self.nSolarPhase[freq] = 0
        u,v,w = base2uvw_36.base2uvw(self.hAngle, self.RAO.declination, 1, 2)
        baseWave = NP.sqrt(u**2+v**2)*self.freqList[freq]*1e3/constants.c.to_value()
        if baseWave > 120:
            self.ewSolarPhase[freq] = NP.pi
        else:
            self.ewSolarPhase[freq] = 0
            
    def calculatePhase_linear(self, freqChannel, baselinesNumber = 1):
        antNumberN = 31
        antNumberEW = 97
        redIndexesN = []
        for baseline in range(1, baselinesNumber+1):
            redIndexesN.append(NP.where((self.antennaA==98-1+baseline) & (self.antennaB==33))[0][0])
            for i in range(antNumberN - baseline):
                redIndexesN.append(NP.where((self.antennaB==98+i) & (self.antennaA==98+i+baseline))[0][0])
    
        redIndexesEW = []
        for baseline in range(1, baselinesNumber+1):
            for i in range(antNumberEW - baseline):
                redIndexesEW.append(NP.where((self.antennaA==1+i) & (self.antennaB==1+i+baseline))[0][0])
                
        phaMatrix = self.phaMatrixGenPairsEWN(baselinesNumber, antNumberEW, antNumberN)
        redundantVisLcp = self.visLcp[freqChannel, self.calibIndex, NP.append(redIndexesEW, redIndexesN)]
        sunVisPhases = NP.zeros(2)
#        if self.freqList[freqChannel] > 4e6:
#            sunVisPhases = NP.array((NP.pi, NP.pi))
        phasesLcp = NP.concatenate((NP.angle(redundantVisLcp), sunVisPhases))
        antPhaLcp, c, d, e = NP.linalg.lstsq(phaMatrix, phasesLcp, rcond=None)
        self.ewAntPhaLcp[freqChannel] = antPhaLcp[baselinesNumber*2:baselinesNumber*2+antNumberEW]
        self.nAntPhaLcp[freqChannel] = antPhaLcp[baselinesNumber*2+antNumberEW:]
        
        redundantVisRcp = self.visRcp[freqChannel, self.calibIndex, NP.append(redIndexesEW, redIndexesN)]
        phasesRcp = NP.concatenate((NP.angle(redundantVisRcp), NP.array((0,0))))
        antPhaRcp, c, d, e = NP.linalg.lstsq(phaMatrix, phasesRcp, rcond=None)
        self.ewAntPhaRcp[freqChannel] = antPhaRcp[baselinesNumber*2:baselinesNumber*2+antNumberEW]
        self.nAntPhaRcp[freqChannel] = antPhaRcp[baselinesNumber*2+antNumberEW:]
        
    def calculatePhaseLcp_nonlinear(self, freqChannel, baselinesNumber = 5):
        antNumberN = 31
        antNumberEW = self.antNumberEW
        redIndexesN = []
        for baseline in range(1, baselinesNumber+1):
            redIndexesN.append(NP.where((self.antennaA==98-1+baseline) & (self.antennaB==33))[0][0])
            for i in range(antNumberN - baseline):
                redIndexesN.append(NP.where((self.antennaB==98+i) & (self.antennaA==98+i+baseline))[0][0])
    
        redIndexesEW = []
        for baseline in range(1, baselinesNumber+1):
            for i in range(antNumberEW - baseline):
                redIndexesEW.append(NP.where((self.antennaA==1+i) & (self.antennaB==1+i+baseline))[0][0])
             
        if self.averageCalib:
            redundantVisN = NP.mean(self.visLcp[freqChannel, :20, redIndexesN], axis = 1)
            redundantVisEW = NP.mean(self.visLcp[freqChannel, :20, redIndexesEW], axis = 1)
            redundantVisAll = NP.append(redundantVisEW, redundantVisN)
        else:
            redundantVisN = self.visLcp[freqChannel, self.calibIndex, redIndexesN]
            redundantVisEW = self.visLcp[freqChannel, self.calibIndex, redIndexesEW]
            redundantVisAll = NP.append(redundantVisEW, redundantVisN)
        
#        x_ini = NP.concatenate((NP.ones(baselinesNumber+antNumberN-1), NP.zeros(baselinesNumber+antNumberN-1)))
#        ls_res = least_squares(self.northGainsFunc_constrained, x_ini, args = (redundantVisN, antNumberN, baselinesNumber, freqChannel), max_nfev = 500)
#        self.n_gains_lcp = self.real_to_complex(ls_res['x'])[baselinesNumber-1:]
#        self.nAntPhaLcp[freqChannel] = NP.angle(self.n_gains_lcp)
#        
#        x_ini = NP.concatenate((NP.ones(baselinesNumber+antNumberEW-2), NP.zeros(baselinesNumber+antNumberEW-2)))
#        ls_res = least_squares(self.eastWestGainsFunc_constrained, x_ini, args = (redundantVisEW, antNumberEW, baselinesNumber, freqChannel), max_nfev = 900)
#        gains = self.real_to_complex(ls_res['x'])[baselinesNumber-1:]
#        self.ew_gains_lcp = NP.insert(gains, 32, (1+0j))
#        self.ewAntPhaLcp[freqChannel] = NP.angle(self.ew_gains_lcp)
        
        
        # x_size = (baselinesNumber-1)*2 + antNumberEW + antNumberN
        # x_ini = NP.concatenate((NP.ones(x_size+1), NP.zeros(x_size)))
        ls_res = least_squares(self.allGainsFunc_constrained, self.x_ini_lcp[freqChannel], args = (redundantVisAll, antNumberEW, antNumberN, baselinesNumber, freqChannel), max_nfev = 400)
        self.calibrationResultLcp[freqChannel] = ls_res['x']
        gains = self.real_to_complex(ls_res['x'][1:])[(baselinesNumber-1)*2:]
        self.ew_gains_lcp = gains[:antNumberEW]
        self.ewAntPhaLcp[freqChannel] = NP.angle(self.ew_gains_lcp)
        self.n_gains_lcp = gains[antNumberEW:]
        self.nAntPhaLcp[freqChannel] = NP.angle(self.n_gains_lcp)
        
        norm = NP.mean(NP.abs(gains))#[NP.abs(gains)<NP.median(NP.abs(gains))*0.6]))
        self.ewAntAmpLcp[freqChannel] = NP.abs(self.ew_gains_lcp)/norm
        self.ewAntAmpLcp[freqChannel][self.ewAntAmpLcp[freqChannel]<NP.median(self.ewAntAmpLcp[freqChannel])*0.6] = 1e6
        self.nAntAmpLcp[freqChannel] = NP.abs(self.n_gains_lcp)/norm
        self.nAntAmpLcp[freqChannel][self.nAntAmpLcp[freqChannel]<NP.median(self.nAntAmpLcp[freqChannel])*0.6] = 1e6
        
    def calculatePhaseLcp_nonlinear_new(self, freqChannel, baselinesNumber = 5):
        antNumberN = 31
        antNumberEW = self.antNumberEW
        redIndexesN = []
        for baseline in range(1, baselinesNumber+1):
            redIndexesN.append(NP.where((self.antennaA==98-1+baseline) & (self.antennaB==33))[0][0])
            for i in range(antNumberN - baseline):
                redIndexesN.append(NP.where((self.antennaB==98+i) & (self.antennaA==98+i+baseline))[0][0])
    
        redIndexesEW = []
        for baseline in range(1, baselinesNumber+1):
            for i in range(antNumberEW - baseline):
                redIndexesEW.append(NP.where((self.antennaA==1+i) & (self.antennaB==1+i+baseline))[0][0])
             
        if self.averageCalib:
            redundantVisN = NP.mean(self.visLcp[freqChannel, :20, redIndexesN], axis = 1)
            redundantVisEW = NP.mean(self.visLcp[freqChannel, :20, redIndexesEW], axis = 1)
            redundantVisAll = NP.append(redundantVisEW, redundantVisN)
        else:
            redundantVisN = self.visLcp[freqChannel, self.calibIndex, redIndexesN]
            redundantVisEW = self.visLcp[freqChannel, self.calibIndex, redIndexesEW]
            redundantVisAll = NP.append(redundantVisEW, redundantVisN)
        
#        x_ini = NP.concatenate((NP.ones(baselinesNumber+antNumberN-1), NP.zeros(baselinesNumber+antNumberN-1)))
#        ls_res = least_squares(self.northGainsFunc_constrained, x_ini, args = (redundantVisN, antNumberN, baselinesNumber, freqChannel), max_nfev = 500)
#        self.n_gains_lcp = self.real_to_complex(ls_res['x'])[baselinesNumber-1:]
#        self.nAntPhaLcp[freqChannel] = NP.angle(self.n_gains_lcp)
#        
#        x_ini = NP.concatenate((NP.ones(baselinesNumber+antNumberEW-2), NP.zeros(baselinesNumber+antNumberEW-2)))
#        ls_res = least_squares(self.eastWestGainsFunc_constrained, x_ini, args = (redundantVisEW, antNumberEW, baselinesNumber, freqChannel), max_nfev = 900)
#        gains = self.real_to_complex(ls_res['x'])[baselinesNumber-1:]
#        self.ew_gains_lcp = NP.insert(gains, 32, (1+0j))
#        self.ewAntPhaLcp[freqChannel] = NP.angle(self.ew_gains_lcp)
        
        
        # x_size = (baselinesNumber-1)*2 + antNumberEW + antNumberN
        # x_ini = NP.concatenate((NP.ones(x_size+1), NP.zeros(x_size)))
        
        ewAmpSign = 1 if self.ewSolarPhase[freqChannel]==0 else -1
        nAmpSign = 1 if self.nSolarPhase[freqChannel]==0 else -1
        
        res = NP.zeros_like(redundantVisAll, dtype = complex)
        ewSolarAmp = 1 * ewAmpSign
        nAntNumber_c = antNumberN + 1
        nGainsNumber = antNumberN
        ewGainsNumber = antNumberEW
        nSolVisNumber = baselinesNumber - 1
        ewSolVisNumber = baselinesNumber - 1
        nNum = int((2*antNumberN - (baselinesNumber-1))/2 * baselinesNumber)
        ewNum = int((2*(antNumberEW-1) - (baselinesNumber-1))/2 * baselinesNumber)
        solVisArrayN = NP.zeros(nNum, dtype = complex)
        antAGainsN = NP.zeros(nNum, dtype = complex)
        antBGainsN = NP.zeros(nNum, dtype = complex)
        solVisArrayEW = NP.zeros(ewNum, dtype = complex)
        antAGainsEW = NP.zeros(ewNum, dtype = complex)
        antBGainsEW = NP.zeros(ewNum, dtype = complex)
        ewSolVis = NP.zeros(baselinesNumber, dtype = complex)
        nSolVis = NP.zeros(baselinesNumber, dtype = complex)
        solVis = NP.zeros_like(redundantVisAll, dtype = complex)
        antAGains = NP.zeros_like(redundantVisAll, dtype = complex)
        antBGains = NP.zeros_like(redundantVisAll, dtype = complex)
        
        args = (redundantVisAll, antNumberEW, antNumberN, baselinesNumber, freqChannel,
                res, ewSolarAmp, nAntNumber_c, nGainsNumber, ewGainsNumber, nSolVisNumber, 
                ewSolVisNumber, nNum, ewNum, solVisArrayN, antAGainsN, antBGainsN, solVisArrayEW, 
                antAGainsEW, antBGainsEW, ewSolVis, nSolVis, solVis, antAGains, antBGains, nAmpSign)
        
        ls_res = least_squares(self.allGainsFunc_constrained_new, self.x_ini_lcp[freqChannel], args = args, max_nfev = 400)
        self.calibrationResultLcp[freqChannel] = ls_res['x']
        gains = self.real_to_complex(ls_res['x'][1:])[(baselinesNumber-1)*2:]
        self.ew_gains_lcp = gains[:antNumberEW]
        self.ewAntPhaLcp[freqChannel] = NP.angle(self.ew_gains_lcp)
        self.n_gains_lcp = gains[antNumberEW:]
        self.nAntPhaLcp[freqChannel] = NP.angle(self.n_gains_lcp)
        
        norm = NP.mean(NP.abs(gains))#[NP.abs(gains)<NP.median(NP.abs(gains))*0.6]))
        self.ewAntAmpLcp[freqChannel] = NP.abs(self.ew_gains_lcp)/norm
        self.ewAntAmpLcp[freqChannel][self.ewAntAmpLcp[freqChannel]<NP.median(self.ewAntAmpLcp[freqChannel])*0.6] = 1e6
        self.nAntAmpLcp[freqChannel] = NP.abs(self.n_gains_lcp)/norm
        self.nAntAmpLcp[freqChannel][self.nAntAmpLcp[freqChannel]<NP.median(self.nAntAmpLcp[freqChannel])*0.6] = 1e6
        
    def calculatePhaseRcp_nonlinear_new(self, freqChannel, baselinesNumber = 5):
        antNumberN = 31
        antNumberEW = self.antNumberEW
        redIndexesN = []
        for baseline in range(1, baselinesNumber+1):
            redIndexesN.append(NP.where((self.antennaA==98-1+baseline) & (self.antennaB==33))[0][0])
            for i in range(antNumberN - baseline):
                redIndexesN.append(NP.where((self.antennaB==98+i) & (self.antennaA==98+i+baseline))[0][0])
    
        redIndexesEW = []
        for baseline in range(1, baselinesNumber+1):
            for i in range(antNumberEW - baseline):
                redIndexesEW.append(NP.where((self.antennaA==1+i) & (self.antennaB==1+i+baseline))[0][0])
             
        if self.averageCalib:
            redundantVisN = NP.mean(self.visRcp[freqChannel, :20, redIndexesN], axis = 1)
            redundantVisEW = NP.mean(self.visRcp[freqChannel, :20, redIndexesEW], axis = 1)
            redundantVisAll = NP.append(redundantVisEW, redundantVisN)
        else:
            redundantVisN = self.visRcp[freqChannel, self.calibIndex, redIndexesN]
            redundantVisEW = self.visRcp[freqChannel, self.calibIndex, redIndexesEW]
            redundantVisAll = NP.append(redundantVisEW, redundantVisN)

        ewAmpSign = 1 if self.ewSolarPhase[freqChannel]==0 else -1
        nAmpSign = 1 if self.nSolarPhase[freqChannel]==0 else -1
        
        res = NP.zeros_like(redundantVisAll, dtype = complex)
        ewSolarAmp = 1 * ewAmpSign
        nAntNumber_c = antNumberN + 1
        nGainsNumber = antNumberN
        ewGainsNumber = antNumberEW
        nSolVisNumber = baselinesNumber - 1
        ewSolVisNumber = baselinesNumber - 1
        nNum = int((2*antNumberN - (baselinesNumber-1))/2 * baselinesNumber)
        ewNum = int((2*(antNumberEW-1) - (baselinesNumber-1))/2 * baselinesNumber)
        solVisArrayN = NP.zeros(nNum, dtype = complex)
        antAGainsN = NP.zeros(nNum, dtype = complex)
        antBGainsN = NP.zeros(nNum, dtype = complex)
        solVisArrayEW = NP.zeros(ewNum, dtype = complex)
        antAGainsEW = NP.zeros(ewNum, dtype = complex)
        antBGainsEW = NP.zeros(ewNum, dtype = complex)
        ewSolVis = NP.zeros(baselinesNumber, dtype = complex)
        nSolVis = NP.zeros(baselinesNumber, dtype = complex)
        solVis = NP.zeros_like(redundantVisAll, dtype = complex)
        antAGains = NP.zeros_like(redundantVisAll, dtype = complex)
        antBGains = NP.zeros_like(redundantVisAll, dtype = complex)
        
        args = (redundantVisAll, antNumberEW, antNumberN, baselinesNumber, freqChannel,
                res, ewSolarAmp, nAntNumber_c, nGainsNumber, ewGainsNumber, nSolVisNumber, 
                ewSolVisNumber, nNum, ewNum, solVisArrayN, antAGainsN, antBGainsN, solVisArrayEW, 
                antAGainsEW, antBGainsEW, ewSolVis, nSolVis, solVis, antAGains, antBGains, nAmpSign)
        
        ls_res = least_squares(self.allGainsFunc_constrained_new, self.x_ini_rcp[freqChannel], args = args, max_nfev = 400)
        self.calibrationResultRcp[freqChannel] = ls_res['x']
        gains = self.real_to_complex(ls_res['x'][1:])[(baselinesNumber-1)*2:]
        self.ew_gains_rcp = gains[:antNumberEW]
        self.ewAntPhaRcp[freqChannel] = NP.angle(self.ew_gains_rcp)
        self.n_gains_rcp = gains[antNumberEW:]
        self.nAntPhaRcp[freqChannel] = NP.angle(self.n_gains_rcp)
        
        norm = NP.mean(NP.abs(gains))#[NP.abs(gains)<NP.median(NP.abs(gains))*0.6]))
        self.ewAntAmpRcp[freqChannel] = NP.abs(self.ew_gains_rcp)/norm
        self.ewAntAmpRcp[freqChannel][self.ewAntAmpRcp[freqChannel]<NP.median(self.ewAntAmpRcp[freqChannel])*0.6] = 1e6
        self.nAntAmpRcp[freqChannel] = NP.abs(self.n_gains_rcp)/norm
        self.nAntAmpRcp[freqChannel][self.nAntAmpRcp[freqChannel]<NP.median(self.nAntAmpRcp[freqChannel])*0.6] = 1e6
        
    def allGainsFunc_constrained_new(self, x, obsVis, ewAntNumber, nAntNumber, baselineNumber, freq, 
                res, ewSolarAmp, nAntNumber_c, nGainsNumber, ewGainsNumber, nSolVisNumber, 
                ewSolVisNumber, nNum, ewNum, solVisArrayN, antAGainsN, antBGainsN, solVisArrayEW, 
                antAGainsEW, antBGainsEW, ewSolVis, nSolVis, solVis, antAGains, antBGains, nAmpSign):
        
        nSolarAmp = NP.abs(x[0]) * nAmpSign
        x_complex = self.real_to_complex(x[1:])
        
        # ewSolVis = NP.append((ewSolarAmp * NP.exp(1j*self.ewSolarPhase[freq])), x_complex[: ewSolVisNumber])
        # nSolVis = NP.append((nSolarAmp * NP.exp(1j*self.nSolarPhase[freq])), x_complex[ewSolVisNumber : ewSolVisNumber+nSolVisNumber])
        
        ewSolVis[0] = ewSolarAmp
        ewSolVis[1:] = x_complex[: ewSolVisNumber]
        nSolVis[0] = nSolarAmp
        nSolVis[1:] = x_complex[ewSolVisNumber : ewSolVisNumber+nSolVisNumber]
        
        ewGains = x_complex[ewSolVisNumber+nSolVisNumber : ewSolVisNumber+nSolVisNumber+ewGainsNumber]
        nGains = NP.append(ewGains[32], x_complex[ewSolVisNumber+nSolVisNumber+ewGainsNumber :])
        
        
        prev_ind_n = 0
        prev_ind_ew = 0
        for baseline in range(1, baselineNumber+1):
            solVisArrayN[prev_ind_n:prev_ind_n+nAntNumber_c-baseline] = NP.full(nAntNumber_c-baseline, nSolVis[baseline-1])
            antAGainsN[prev_ind_n:prev_ind_n+nAntNumber_c-baseline] = nGains[:nAntNumber_c-baseline]
            antBGainsN[prev_ind_n:prev_ind_n+nAntNumber_c-baseline] = nGains[baseline:]
            prev_ind_n = prev_ind_n+nAntNumber_c-baseline
            
            solVisArrayEW[prev_ind_ew:prev_ind_ew+ewAntNumber-baseline] = NP.full(ewAntNumber-baseline, ewSolVis[baseline-1])
            antAGainsEW[prev_ind_ew:prev_ind_ew+ewAntNumber-baseline] = ewGains[:ewAntNumber-baseline]
            antBGainsEW[prev_ind_ew:prev_ind_ew+ewAntNumber-baseline] = ewGains[baseline:]
            prev_ind_ew = prev_ind_ew+ewAntNumber-baseline
            
        solVis[:len(solVisArrayEW)] = solVisArrayEW
        solVis[len(solVisArrayEW):] = solVisArrayN
        antAGains[:len(antAGainsEW)] = antAGainsEW
        antAGains[len(antAGainsEW):] = antAGainsN
        antBGains[:len(antBGainsEW)] = antBGainsEW
        antBGains[len(antBGainsEW):] = antBGainsN
            
        res = solVis * antAGains * NP.conj(antBGains) - obsVis
        return self.complex_to_real(res)  
        
    def calculatePhaseRcp_nonlinear(self, freqChannel, baselinesNumber = 5):
        antNumberN = 31
        antNumberEW = self.antNumberEW
        redIndexesN = []
        for baseline in range(1, baselinesNumber+1):
            redIndexesN.append(NP.where((self.antennaA==98-1+baseline) & (self.antennaB==33))[0][0])
            for i in range(antNumberN - baseline):
                redIndexesN.append(NP.where((self.antennaB==98+i) & (self.antennaA==98+i+baseline))[0][0])
    
        redIndexesEW = []
        for baseline in range(1, baselinesNumber+1):
            for i in range(antNumberEW - baseline):
                redIndexesEW.append(NP.where((self.antennaA==1+i) & (self.antennaB==1+i+baseline))[0][0])
             
        if self.averageCalib:
            redundantVisN = NP.mean(self.visRcp[freqChannel, :20, redIndexesN], axis = 1)
            redundantVisEW = NP.mean(self.visRcp[freqChannel, :20, redIndexesEW], axis = 1)
            redundantVisAll = NP.append(redundantVisEW, redundantVisN)
        else:
            redundantVisN = self.visRcp[freqChannel, self.calibIndex, redIndexesN]
            redundantVisEW = self.visRcp[freqChannel, self.calibIndex, redIndexesEW]
            redundantVisAll = NP.append(redundantVisEW, redundantVisN)
        
#        x_ini = NP.concatenate((NP.ones(baselinesNumber+antNumberN-1), NP.zeros(baselinesNumber+antNumberN-1)))
#        ls_res = least_squares(self.northGainsFunc_constrained, x_ini, args = (redundantVisN, antNumberN, baselinesNumber, freqChannel), max_nfev = 500)
#        self.n_gains_rcp = self.real_to_complex(ls_res['x'])[baselinesNumber-1:]
#        self.nAntPhaRcp[freqChannel] = NP.angle(self.n_gains_rcp)
#        
#        x_ini = NP.concatenate((NP.ones(baselinesNumber+antNumberEW-2), NP.zeros(baselinesNumber+antNumberEW-2)))
#        ls_res = least_squares(self.eastWestGainsFunc_constrained, x_ini, args = (redundantVisEW, antNumberEW, baselinesNumber, freqChannel), max_nfev = 900)
#        gains = self.real_to_complex(ls_res['x'])[baselinesNumber-1:]
#        self.ew_gains_rcp = NP.insert(gains, 32, (1+0j))
#        self.ewAntPhaRcp[freqChannel] = NP.angle(self.ew_gains_rcp)
        
        # x_size = (baselinesNumber-1)*2 + antNumberEW + antNumberN
        # self.x_ini = NP.concatenate((NP.ones(x_size+1), NP.zeros(x_size)))
        ls_res = least_squares(self.allGainsFunc_constrained, self.x_ini_rcp[freqChannel], args = (redundantVisAll, antNumberEW, antNumberN, baselinesNumber, freqChannel), max_nfev = 400)
        self.calibrationResultRcp[freqChannel] = ls_res['x']
        gains = self.real_to_complex(ls_res['x'][1:])[(baselinesNumber-1)*2:]
        self.ew_gains_rcp = gains[:antNumberEW]
        self.ewAntPhaRcp[freqChannel] = NP.angle(self.ew_gains_rcp)
        self.n_gains_rcp = gains[antNumberEW:]
        self.nAntPhaRcp[freqChannel] = NP.angle(self.n_gains_rcp)
        
        norm = NP.mean(NP.abs(gains))#[NP.abs(gains)<NP.median(NP.abs(gains))*0.6]))
        self.ewAntAmpRcp[freqChannel] = NP.abs(self.ew_gains_rcp)/norm
        self.ewAntAmpRcp[freqChannel][self.ewAntAmpRcp[freqChannel]<NP.median(self.ewAntAmpRcp[freqChannel])*0.6] = 1e6
        self.nAntAmpRcp[freqChannel] = NP.abs(self.n_gains_rcp)/norm
        self.nAntAmpRcp[freqChannel][self.nAntAmpRcp[freqChannel]<NP.median(self.nAntAmpRcp[freqChannel])*0.6] = 1e6
     
    def calculateAmplitude_linear(self, freqChannel, baselinesNumber = 3):    
        antNumberN = 31
        antNumberEW = self.antNumberEW
        redIndexesN = []
        for baseline in range(1, baselinesNumber+1):
            redIndexesN.append(NP.where((self.antennaA==98-1+baseline) & (self.antennaB==33))[0][0])
            for i in range(antNumberN - baseline):
                redIndexesN.append(NP.where((self.antennaB==98+i) & (self.antennaA==98+i+baseline))[0][0])
    
        redIndexesEW = []
        for baseline in range(1, baselinesNumber+1):
            for i in range(antNumberEW - baseline):
                redIndexesEW.append(NP.where((self.antennaA==1+i) & (self.antennaB==1+i+baseline))[0][0])
             
        if self.averageCalib:
            redundantVisN = NP.mean(self.visLcp[freqChannel, :20, redIndexesN], axis = 1)
            redundantVisEW = NP.mean(self.visLcp[freqChannel, :20, redIndexesEW], axis = 1)
            redundantVisAllLcp = NP.append(redundantVisEW, redundantVisN)
            redundantVisN = NP.mean(self.visRcp[freqChannel, :20, redIndexesN], axis = 1)
            redundantVisEW = NP.mean(self.visRcp[freqChannel, :20, redIndexesEW], axis = 1)
            redundantVisAllRcp = NP.append(redundantVisEW, redundantVisN)
        else:
            redundantVisN = self.visLcp[freqChannel, self.calibIndex, redIndexesN]
            redundantVisEW = self.visLcp[freqChannel, self.calibIndex, redIndexesEW]
            redundantVisAllLcp = NP.append(redundantVisEW, redundantVisN)
            redundantVisN = self.visRcp[freqChannel, self.calibIndex, redIndexesN]
            redundantVisEW = self.visRcp[freqChannel, self.calibIndex, redIndexesEW]
            redundantVisAllRcp = NP.append(redundantVisEW, redundantVisN)
            
        ampMatrix = NP.abs(self.phaMatrixGenPairsEWN(baselinesNumber, antNumberEW, antNumberN))
        
        allAmp = NP.abs(redundantVisAllLcp)
        antAmp, c, d, e = NP.linalg.lstsq(ampMatrix,NP.log(allAmp), rcond=None)
        antAmp= NP.exp(antAmp[baselinesNumber*2:])
        self.ewAntAmpLcp[freqChannel] = antAmp[:antNumberEW]
        self.nAntAmpLcp[freqChannel] = antAmp[antNumberEW:]
        self.ewAntAmpLcp[freqChannel][self.ewAntAmpLcp[freqChannel]<NP.median(self.ewAntAmpLcp[freqChannel])*0.6] = 1e6
        self.nAntAmpLcp[freqChannel][self.nAntAmpLcp[freqChannel]<NP.median(self.nAntAmpLcp[freqChannel])*0.6] = 1e6
        
        allAmp = NP.abs(redundantVisAllRcp)
        antAmp, c, d, e = NP.linalg.lstsq(ampMatrix,NP.log(allAmp), rcond=None)
        antAmp= NP.exp(antAmp[baselinesNumber*2:])
        self.ewAntAmpRcp[freqChannel] = antAmp[:antNumberEW]
        self.nAntAmpRcp[freqChannel] = antAmp[antNumberEW:]
        self.ewAntAmpRcp[freqChannel][self.ewAntAmpRcp[freqChannel]<NP.median(self.ewAntAmpRcp[freqChannel])*0.6] = 1e6
        self.nAntAmpRcp[freqChannel][self.nAntAmpRcp[freqChannel]<NP.median(self.nAntAmpRcp[freqChannel])*0.6] = 1e6
    
    def eastWestGainsFunc_constrained(self, x, obsVis, antNumber, baselineNumber, freq):
        res = NP.zeros_like(obsVis, dtype = complex)
        x_complex = self.real_to_complex(x)
        solVis = NP.append((NP.exp(1j*self.ewSolarPhase[freq])), x_complex[:baselineNumber-1])
        gains = NP.insert(x_complex[baselineNumber-1:], 32, (1+0j))
        
        solVisArray = NP.array(())
        antAGains = NP.array(())
        antBGains = NP.array(())
        for baseline in range(1, baselineNumber+1):
            solVisArray = NP.append(solVisArray, NP.full(antNumber-baseline, solVis[baseline-1]))
            antAGains = NP.append(antAGains, gains[:antNumber-baseline])
            antBGains = NP.append(antBGains, gains[baseline:])
        res = solVisArray * antAGains * NP.conj(antBGains) - obsVis
        return self.complex_to_real(res)

    def northGainsFunc_constrained(self, x, obsVis, antNumber, baselineNumber, freq):
        antNumber_c = antNumber + 1
        res = NP.zeros_like(obsVis, dtype = complex)
        x_complex = self.real_to_complex(x)
        solVis = NP.append((NP.exp(1j*self.nSolarPhase[freq])), x_complex[:baselineNumber-1])
        gains = NP.append((1+0j), x_complex[baselineNumber-1:])
        
        solVisArray = NP.array(())
        antAGains = NP.array(())
        antBGains = NP.array(())
        for baseline in range(1, baselineNumber+1):
            solVisArray = NP.append(solVisArray, NP.full(antNumber_c-baseline, solVis[baseline-1]))
            antAGains = NP.append(antAGains, gains[:antNumber_c-baseline])
            antBGains = NP.append(antBGains, gains[baseline:])
        res = solVisArray * antAGains * NP.conj(antBGains) - obsVis
        return self.complex_to_real(res)
    
    def allGainsFunc_constrained(self, x, obsVis, ewAntNumber, nAntNumber, baselineNumber, freq):
        res = NP.zeros_like(obsVis, dtype = complex)
        ewSolarAmp = 1
        nSolarAmp = NP.abs(x[0])
        x_complex = self.real_to_complex(x[1:])
        
        nAntNumber_c = nAntNumber + 1
        
        nGainsNumber = nAntNumber
        ewGainsNumber = ewAntNumber
        nSolVisNumber = baselineNumber - 1
        ewSolVisNumber = baselineNumber - 1
        ewSolVis = NP.append((ewSolarAmp * NP.exp(1j*self.ewSolarPhase[freq])), x_complex[: ewSolVisNumber])
        nSolVis = NP.append((nSolarAmp * NP.exp(1j*self.nSolarPhase[freq])), x_complex[ewSolVisNumber : ewSolVisNumber+nSolVisNumber])
        ewGains = x_complex[ewSolVisNumber+nSolVisNumber : ewSolVisNumber+nSolVisNumber+ewGainsNumber]
        nGains = NP.append(ewGains[32], x_complex[ewSolVisNumber+nSolVisNumber+ewGainsNumber :])
        
        solVisArrayN = NP.array(())
        antAGainsN = NP.array(())
        antBGainsN = NP.array(())
        solVisArrayEW = NP.array(())
        antAGainsEW = NP.array(())
        antBGainsEW = NP.array(())
        for baseline in range(1, baselineNumber+1):
            solVisArrayN = NP.append(solVisArrayN, NP.full(nAntNumber_c-baseline, nSolVis[baseline-1]))
            antAGainsN = NP.append(antAGainsN, nGains[:nAntNumber_c-baseline])
            antBGainsN = NP.append(antBGainsN, nGains[baseline:])
            
            solVisArrayEW = NP.append(solVisArrayEW, NP.full(ewAntNumber-baseline, ewSolVis[baseline-1]))
            antAGainsEW = NP.append(antAGainsEW, ewGains[:ewAntNumber-baseline])
            antBGainsEW = NP.append(antBGainsEW, ewGains[baseline:])
            
        res = NP.append(solVisArrayEW, solVisArrayN) * NP.append(antAGainsEW, antAGainsN) * NP.conj(NP.append(antBGainsEW, antBGainsN)) - obsVis
        return self.complex_to_real(res)  
    
    def allGainsFunc_constrained_old(self, x, obsVis, ewAntNumber, nAntNumber, baselineNumber, freq):
        res = NP.zeros_like(obsVis, dtype = complex)
        ewSolarAmp = NP.abs(x[0])
        nSolarAmp = NP.abs(x[1])
        x_complex = self.real_to_complex(x[2:])
        
        nAntNumber_c = nAntNumber + 1
        
        nGainsNumber = nAntNumber
        ewGainsNumber = ewAntNumber
        nSolVisNumber = baselineNumber - 1
        ewSolVisNumber = baselineNumber - 1
        ewSolVis = NP.append((ewSolarAmp * NP.exp(1j*self.ewSolarPhase[freq])), x_complex[: ewSolVisNumber])
        nSolVis = NP.append((nSolarAmp * NP.exp(1j*self.nSolarPhase[freq])), x_complex[ewSolVisNumber : ewSolVisNumber+nSolVisNumber])
        ewGains = x_complex[ewSolVisNumber+nSolVisNumber : ewSolVisNumber+nSolVisNumber+ewGainsNumber]
        nGains = NP.append(ewGains[32], x_complex[ewSolVisNumber+nSolVisNumber+ewGainsNumber :])
        
        solVisArrayN = NP.array(())
        antAGainsN = NP.array(())
        antBGainsN = NP.array(())
        solVisArrayEW = NP.array(())
        antAGainsEW = NP.array(())
        antBGainsEW = NP.array(())
        for baseline in range(1, baselineNumber+1):
            solVisArrayN = NP.append(solVisArrayN, NP.full(nAntNumber_c-baseline, nSolVis[baseline-1]))
            antAGainsN = NP.append(antAGainsN, nGains[:nAntNumber_c-baseline])
            antBGainsN = NP.append(antBGainsN, nGains[baseline:])
            
            solVisArrayEW = NP.append(solVisArrayEW, NP.full(ewAntNumber-baseline, ewSolVis[baseline-1]))
            antAGainsEW = NP.append(antAGainsEW, ewGains[:ewAntNumber-baseline])
            antBGainsEW = NP.append(antBGainsEW, ewGains[baseline:])
            
        res = NP.append(solVisArrayEW, solVisArrayN) * NP.append(antAGainsEW, antAGainsN) * NP.conj(NP.append(antBGainsEW, antBGainsN)) - obsVis
        return self.complex_to_real(res)  
    
    def correctPhaseSlopeRL(self, freq):
        workingAnts_ew = NP.arange(0,97,1)
        workingAnts_ew = NP.delete(workingAnts_ew, NP.append(self.flags_ew, NP.array((28,))))
        phaseDif_ew = NP.unwrap((self.ewAntPhaLcp[freq][workingAnts_ew]+self.ewLcpPhaseCorrection[freq][workingAnts_ew])
                             - (self.ewAntPhaRcp[freq][workingAnts_ew]+self.ewRcpPhaseCorrection[freq][workingAnts_ew]))
        A = NP.vstack([workingAnts_ew, NP.ones(len(workingAnts_ew))]).T
        ew_slope, c = NP.linalg.lstsq(A, phaseDif_ew, rcond=None)[0]
        workingAnts_n = NP.arange(0,31,1)
        workingAnts_n = NP.delete(workingAnts_n, self.flags_n)
        phaseDif_n = NP.unwrap((self.nAntPhaLcp[freq][workingAnts_n]+self.nLcpPhaseCorrection[freq][workingAnts_n])
                             - (self.nAntPhaRcp[freq][workingAnts_n]+self.nRcpPhaseCorrection[freq][workingAnts_n]))
        A = NP.vstack([workingAnts_n, NP.ones(len(workingAnts_n))]).T
        n_slope, c = NP.linalg.lstsq(A, phaseDif_n, rcond=None)[0]
        self.ewSlopeRcp[freq] = self.wrap(self.ewSlopeRcp[freq] + NP.rad2deg(ew_slope))
        self.nSlopeRcp[freq] = self.wrap(self.nSlopeRcp[freq] - NP.rad2deg(n_slope))
    
    def buildEwPhase(self):
        newLcpPhaseCorrection = NP.zeros(self.antNumberEW)
        newRcpPhaseCorrection = NP.zeros(self.antNumberEW)
        for j in range(self.antNumberEW):
                newLcpPhaseCorrection[j] += NP.deg2rad(self.ewSlopeLcp[self.frequencyChannel] * (j - 32)) 
                newRcpPhaseCorrection[j] += NP.deg2rad(self.ewSlopeRcp[self.frequencyChannel] * (j - 32))
        self.ewLcpPhaseCorrection[self.frequencyChannel, :] = newLcpPhaseCorrection[:]
        self.ewRcpPhaseCorrection[self.frequencyChannel, :] = newRcpPhaseCorrection[:]
        
    def buildNPhase(self):
        newLcpPhaseCorrection = NP.zeros(self.antNumberN)
        newRcpPhaseCorrection = NP.zeros(self.antNumberN)
        for j in range(self.antNumberN):
                newLcpPhaseCorrection[j] += (NP.deg2rad(-self.nSlopeLcp[self.frequencyChannel] * (j + 1))  + NP.deg2rad(self.sLcpStair[self.frequencyChannel]))
                newRcpPhaseCorrection[j] += (NP.deg2rad(-self.nSlopeRcp[self.frequencyChannel] * (j + 1)) + NP.deg2rad(self.sRcpStair[self.frequencyChannel]))
        self.nLcpPhaseCorrection[self.frequencyChannel, :] = newLcpPhaseCorrection[:]
        self.nRcpPhaseCorrection[self.frequencyChannel, :] = newRcpPhaseCorrection[:]
    
    def real_to_complex(self, z):
        return z[:len(z)//2] + 1j * z[len(z)//2:]
    
    def complex_to_real(self, z):
        return NP.concatenate((NP.real(z), NP.imag(z)))
    
    def setCalibIndex(self, calibIndex):
        self.calibIndex = calibIndex;

    def setFrequencyChannel(self, channel):
        self.frequencyChannel = channel
        
    def vis2uv(self, scan, phaseCorrect = True, amplitudeCorrect = False, PSF=False, average = 0):
        self.uvLcp[:,:] = complex(0,0)
        self.uvRcp[:,:] = complex(0,0)
        flags_ew = NP.where(self.ewAntAmpLcp[self.frequencyChannel]==1e6)[0]
        flags_n = NP.where(self.nAntAmpLcp[self.frequencyChannel]==1e6)[0]
        
        ewPhLcp = self.ewAntPhaLcp[self.frequencyChannel] + self.ewLcpPhaseCorrection[self.frequencyChannel]
        nPhLcp = self.nAntPhaLcp[self.frequencyChannel] + self.nLcpPhaseCorrection[self.frequencyChannel]
        ewAmpLcp = self.ewAntAmpLcp[self.frequencyChannel]
        nAmpLcp = self.nAntAmpLcp[self.frequencyChannel]
        if self.useRLDif:
            ewPhRcp = ewPhLcp - self.ewPhaseDif[self.frequencyChannel]
            nPhRcp = nPhLcp - self.nPhaseDif[self.frequencyChannel]
            ewAmpRcp = ewAmpLcp/self.ewAmpDif[self.frequencyChannel]
            nAmpRcp = nAmpLcp/self.nAmpDif[self.frequencyChannel]
        else:
            ewPhRcp = self.ewAntPhaRcp[self.frequencyChannel] + self.ewRcpPhaseCorrection[self.frequencyChannel]
            nPhRcp = self.nAntPhaRcp[self.frequencyChannel] + self.nRcpPhaseCorrection[self.frequencyChannel]
            ewAmpRcp = self.ewAntAmpRcp[self.frequencyChannel]
            nAmpRcp = self.nAntAmpRcp[self.frequencyChannel]
        
        O = self.sizeOfUv//2
        if average:
            firstScan = scan
            if  self.visLcp.shape[1] < (scan + average):
                lastScan = self.dataLength
            else:
                lastScan = scan + average
            for i in range(31):
                for j in range(self.antNumberEW):
                    if not (NP.any(flags_ew == j) or NP.any(flags_n == i)):
                        self.uvLcp[O + (i+1)*2, O + (j-32)*2] = NP.mean(self.visLcp[self.frequencyChannel, firstScan:lastScan, i*97+j])
                        self.uvRcp[O + (i+1)*2, O + (j-32)*2] = NP.mean(self.visRcp[self.frequencyChannel, firstScan:lastScan, i*97+j])
                        if (phaseCorrect):
                            self.uvLcp[O + (i+1)*2, O + (j-32)*2] *= NP.exp(1j * (-ewPhLcp[j] + nPhLcp[i]))
                            self.uvRcp[O + (i+1)*2, O + (j-32)*2] *= NP.exp(1j * (-ewPhRcp[j] + nPhRcp[i]))
                        if (amplitudeCorrect):
                            self.uvLcp[O + (i+1)*2, O + (j-32)*2] /= (ewAmpLcp[j] * nAmpLcp[i])
                            self.uvRcp[O + (i+1)*2, O + (j-32)*2] /= (ewAmpRcp[j] * nAmpRcp[i])
                        self.uvLcp[O - (i+1)*2, O - (j-32)*2] = NP.conj(self.uvLcp[O + (i+1)*2, O + (j-32)*2])
                        self.uvRcp[O - (i+1)*2, O - (j-32)*2] = NP.conj(self.uvRcp[O + (i+1)*2, O + (j-32)*2])
            for i in range(self.antNumberEW):
                if not (NP.any(flags_ew == i) or NP.any(flags_ew == 32)):
                    if i<32:
                        self.uvLcp[O, O + (i-32)*2] = NP.mean(self.visLcp[self.frequencyChannel, firstScan:lastScan, self.antZeroRow[i]])
                        self.uvRcp[O, O + (i-32)*2] = NP.mean(self.visRcp[self.frequencyChannel, firstScan:lastScan, self.antZeroRow[i]])
                        if (phaseCorrect):
                            self.uvLcp[O, O + (i-32)*2] *= NP.exp(1j * (-ewPhLcp[i] + ewPhLcp[32]))
                            self.uvRcp[O, O + (i-32)*2] *= NP.exp(1j * (-ewPhRcp[i] + ewPhRcp[32]))
                        if (amplitudeCorrect):
                            self.uvLcp[O, O + (i-32)*2] /= (ewAmpLcp[i] * ewAmpLcp[32])
                            self.uvRcp[O, O + (i-32)*2] /= (ewAmpRcp[i] * ewAmpRcp[32])
    #                    self.uvLcp[O, O + (32-i)*2] = NP.conj(self.uvLcp[O, O + (i-32)*2])
    #                    self.uvRcp[O, O + (32-i)*2] = NP.conj(self.uvRcp[O, O + (i-32)*2])
                    if i>32:
                        self.uvLcp[O, O + (i-32)*2] = NP.conj(NP.mean(self.visLcp[self.frequencyChannel, firstScan:lastScan, self.antZeroRow[i]]))
                        self.uvRcp[O, O + (i-32)*2] = NP.conj(NP.mean(self.visRcp[self.frequencyChannel, firstScan:lastScan, self.antZeroRow[i]]))
                        if (phaseCorrect):
                            self.uvLcp[O, O + (i-32)*2] *= NP.exp(1j * (-ewPhLcp[i] + ewPhLcp[32]))
                            self.uvRcp[O, O + (i-32)*2] *= NP.exp(1j * (-ewPhRcp[i] + ewPhRcp[32]))
                        if (amplitudeCorrect):
                            self.uvLcp[O, O + (i-32)*2] /= (ewAmpLcp[i] * ewAmpLcp[32])
                            self.uvRcp[O, O + (i-32)*2] /= (ewAmpRcp[i] * ewAmpRcp[32])
    #                    self.uvLcp[O, O + (32-i)*2] = NP.conj(self.uvLcp[O, O + (i-32)*2])
    #                    self.uvRcp[O, O + (32-i)*2] = NP.conj(self.uvRcp[O, O + (i-32)*2])
                
        else:
            for i in range(31):
                for j in range(self.antNumberEW):
                    if not (NP.any(flags_ew == j) or NP.any(flags_n == i)):
                        self.uvLcp[O + (i+1)*2, O + (j-32)*2] = self.visLcp[self.frequencyChannel, scan, i*97+j]
                        self.uvRcp[O + (i+1)*2, O + (j-32)*2] = self.visRcp[self.frequencyChannel, scan, i*97+j]
                        if (phaseCorrect):
                            self.uvLcp[O + (i+1)*2, O + (j-32)*2] *= NP.exp(1j * (-ewPhLcp[j] + nPhLcp[i]))
                            self.uvRcp[O + (i+1)*2, O + (j-32)*2] *= NP.exp(1j * (-ewPhRcp[j] + nPhRcp[i]))
                        if (amplitudeCorrect):
                            self.uvLcp[O + (i+1)*2, O + (j-32)*2] /= (ewAmpLcp[j] * nAmpLcp[i])
                            self.uvRcp[O + (i+1)*2, O + (j-32)*2] /= (ewAmpRcp[j] * nAmpRcp[i])
                        if (self.fringeStopping):
                            self.uvLcp[O + (i+1)*2, O + (j-32)*2] *= NP.exp(1j * 2*NP.pi*self.freqList[self.frequencyChannel]*1e3 * (-self.nDelays[i, scan] + self.ewDelays[j, scan]))
                            self.uvRcp[O + (i+1)*2, O + (j-32)*2] *= NP.exp(1j * 2*NP.pi*self.freqList[self.frequencyChannel]*1e3 * (-self.nDelays[i, scan] + self.ewDelays[j, scan]))
                        self.uvLcp[O - (i+1)*2, O - (j-32)*2] = NP.conj(self.uvLcp[O + (i+1)*2, O + (j-32)*2])
                        self.uvRcp[O - (i+1)*2, O - (j-32)*2] = NP.conj(self.uvRcp[O + (i+1)*2, O + (j-32)*2])
                    
            for i in range(self.antNumberEW):
                if not (NP.any(flags_ew == i) or NP.any(flags_ew == 32)):
                    if i<32:
                        self.uvLcp[O, O + (i-32)*2] = self.visLcp[self.frequencyChannel, scan, self.antZeroRow[i]]
                        self.uvRcp[O, O + (i-32)*2] = self.visRcp[self.frequencyChannel, scan, self.antZeroRow[i]]
                        if (phaseCorrect):
                            self.uvLcp[O, O + (i-32)*2] *= NP.exp(1j * (-ewPhLcp[i] + ewPhLcp[32]))
                            self.uvRcp[O, O + (i-32)*2] *= NP.exp(1j * (-ewPhRcp[i] + ewPhRcp[32]))
                        if (amplitudeCorrect):
                            self.uvLcp[O, O + (i-32)*2] /= (ewAmpLcp[i] * ewAmpLcp[32])
                            self.uvRcp[O, O + (i-32)*2] /= (ewAmpRcp[i] * ewAmpRcp[32])
                        if (self.fringeStopping):
                            self.uvLcp[O, O + (i-32)*2] *= NP.exp(1j * 2*NP.pi*self.freqList[self.frequencyChannel]*1e3 * (self.ewDelays[32, scan] + self.ewDelays[i, scan]))
                            self.uvRcp[O, O + (i-32)*2] *= NP.exp(1j * 2*NP.pi*self.freqList[self.frequencyChannel]*1e3 * (self.ewDelays[32, scan] + self.ewDelays[i, scan]))
    #                    self.uvLcp[O, O + (32-i)*2] = NP.conj(self.uvLcp[O, O + (i-32)*2])
    #                    self.uvRcp[O, O + (32-i)*2] = NP.conj(self.uvRcp[O, O + (i-32)*2])
                    if i>32:
                        self.uvLcp[O, O + (i-32)*2] = NP.conj(self.visLcp[self.frequencyChannel, scan, self.antZeroRow[i]])
                        self.uvRcp[O, O + (i-32)*2] = NP.conj(self.visRcp[self.frequencyChannel, scan, self.antZeroRow[i]])
                        if (phaseCorrect):
                            self.uvLcp[O, O + (i-32)*2] *= NP.exp(1j * (-ewPhLcp[i] + ewPhLcp[32]))
                            self.uvRcp[O, O + (i-32)*2] *= NP.exp(1j * (-ewPhRcp[i] + ewPhRcp[32]))
                        if (amplitudeCorrect):
                            self.uvLcp[O, O + (i-32)*2] /= (ewAmpLcp[i] * ewAmpLcp[32])
                            self.uvRcp[O, O + (i-32)*2] /= (ewAmpRcp[i] * ewAmpRcp[32])
                        if (self.fringeStopping):
                            self.uvLcp[O, O + (i-32)*2] *= NP.exp(1j * 2*NP.pi*self.freqList[self.frequencyChannel]*1e3 * (self.ewDelays[32, scan] + self.ewDelays[i, scan]))
                            self.uvRcp[O, O + (i-32)*2] *= NP.exp(1j * 2*NP.pi*self.freqList[self.frequencyChannel]*1e3 * (self.ewDelays[32, scan] + self.ewDelays[i, scan]))
                        # self.uvLcp[O, O + (32-i)*2] = NP.conj(self.uvLcp[O, O + (i-32)*2])
                        # self.uvRcp[O, O + (32-i)*2] = NP.conj(self.uvRcp[O, O + (i-32)*2])
        if (amplitudeCorrect):
            # self.uvLcp[O,O] = self.lcpShift[self.frequencyChannel]
            # self.uvRcp[O,O] = self.rcpShift[self.frequencyChannel]
            self.uvLcp[O,O] = self.fluxLcp[self.frequencyChannel]
            self.uvRcp[O,O] = self.fluxRcp[self.frequencyChannel]
        
        if PSF:
            self.uvLcp[NP.abs(self.uvLcp)>1e-8] = 1
            self.uvRcp[NP.abs(self.uvRcp)>1e-8] = 1
            
        self.uvLcp[NP.abs(self.uvLcp)<1e-6] = 0.
        self.uvRcp[NP.abs(self.uvRcp)<1e-6] = 0.
        self.uvLcp /= NP.count_nonzero(self.uvLcp)
        self.uvRcp /= NP.count_nonzero(self.uvRcp)

    def uv2lmImage(self):
        self.lcp = NP.fft.fft2(NP.roll(NP.roll(self.uvLcp,self.sizeOfUv//2+1,0),self.sizeOfUv//2+1,1));
        self.lcp = NP.roll(NP.roll(self.lcp,self.sizeOfUv//2-1,0),self.sizeOfUv//2-1,1);
        self.lcp = NP.flip(self.lcp, 1)
        self.rcp = NP.fft.fft2(NP.roll(NP.roll(self.uvRcp,self.sizeOfUv//2+1,0),self.sizeOfUv//2+1,1));
        self.rcp = NP.roll(NP.roll(self.rcp,self.sizeOfUv//2-1,0),self.sizeOfUv//2-1,1);
        self.rcp = NP.flip(self.rcp, 1)
        if self.flux_calibrated:
            lam = scipy.constants.c/(self.freqList[self.frequencyChannel]*1e3)
            self.lcp = self.lcp * lam**2 * 1e-22 / (2*scipy.constants.k * self.beam_sr[self.frequencyChannel])
            self.rcp = self.rcp * lam**2 * 1e-22 / (2*scipy.constants.k * self.beam_sr[self.frequencyChannel])
        
    def lm2Heliocentric(self, image_scale = 0.5):
        scaling = self.getPQScale(self.sizeOfUv, NP.deg2rad(self.arcsecPerPixel * (self.sizeOfUv - 1)/3600.)/image_scale)
        # scaling = self.getPQScale(self.sizeOfUv, NP.deg2rad(self.arcsecPerPixel * (self.sizeOfUv - 1)/3600.)*2)
        scale = AffineTransform(scale=(self.sizeOfUv/scaling[0], self.sizeOfUv/scaling[1]))
        shift = AffineTransform(translation=(-self.sizeOfUv/2,-self.sizeOfUv/2))
        rotate = AffineTransform(rotation = self.pAngle)
        matrix = AffineTransform(matrix = self.getPQ2HDMatrix())
        back_shift = AffineTransform(translation=(self.sizeOfUv/2,self.sizeOfUv/2))

        O = self.sizeOfUv//2
        Q = self.sizeOfUv//4
        dataResult0 = warp(self.lcp.real,(shift + (scale + back_shift)).inverse)
        self.lcp = warp(dataResult0,(shift + (matrix + back_shift)).inverse)
        dataResult0 = warp(self.rcp.real,(shift + (scale + back_shift)).inverse)
        self.rcp = warp(dataResult0,(shift + (matrix + back_shift)).inverse)
        dataResult0 = 0
        self.lcp = warp(self.lcp,(shift + (rotate + back_shift)).inverse)[O-Q:O+Q,O-Q:O+Q]
        self.rcp = warp(self.rcp,(shift + (rotate + back_shift)).inverse)[O-Q:O+Q,O-Q:O+Q]

    def calculateDelays(self):
        self.nDelays = NP.zeros((self.antNumberN, self.dataLength))
        self.ewDelays = NP.zeros((self.antNumberEW, self.dataLength))
        base = 9.8
        ew_center = 32
        ns_center = 0
        _PHI = 0.903338787600965
        for scan in range(self.dataLength):
            hourAngle = self.omegaEarth * (self.freqTime[self.frequencyChannel, scan] - self.RAO.culmination)
            dec = self.getDeclination()
            for ant in range(self.antNumberN):
                cosQ = NP.cos(hourAngle) * NP.cos(dec)*NP.sin(_PHI) - NP.sin(dec)*NP.cos(_PHI);
                M = ns_center - ant - 1;
                self.nDelays[ant, scan] = base * M * cosQ / constants.c.to_value();
            for ant in range(self.antNumberEW):
                cosP = NP.sin(hourAngle) * NP.cos(dec);
                N = ew_center - ant;
                self.ewDelays[ant, scan] = base * N * cosP / constants.c.to_value();
                
    def resetDelays(self):
        self.nDelays = NP.zeros((self.antNumberN, self.dataLength))
        self.ewDelays = NP.zeros((self.antNumberEW, self.dataLength))
        
    def wrap(self, value):
        while value<-180:
            value+=360
        while value>180:
            value-=360
        return value
    
    def createDisk(self, radius, arcsecPerPixel = 2.45552):
        qSun = NP.zeros((self.sizeOfUv, self.sizeOfUv))
        sunRadius = radius / (arcsecPerPixel)
        for i in range(self.sizeOfUv):
            x = i - self.sizeOfUv//2 - 1
            for j in range(self.sizeOfUv):
                y = j - self.sizeOfUv//2 - 1
                if (NP.sqrt(x*x + y*y) < sunRadius):
                    qSun[i , j] = 1
                    
        dL = 2*( 30//2) + 1
        arg_x = NP.linspace(-1.,1,dL)
        arg_y = NP.linspace(-1.,1,dL)
        xx, yy = NP.meshgrid(arg_x, arg_y)
        
        scaling = self.getPQScale(self.sizeOfUv, NP.deg2rad(arcsecPerPixel*(self.sizeOfUv-1)/3600.))
        scale = AffineTransform(scale=(scaling[0]/self.sizeOfUv, scaling[1]/self.sizeOfUv))
        back_shift = AffineTransform(translation=(self.sizeOfUv/2, self.sizeOfUv/2))
        shift = AffineTransform(translation=(-self.sizeOfUv/2, -self.sizeOfUv/2))
        matrix = AffineTransform(matrix = NP.linalg.inv(self.getPQ2HDMatrix()))
        rotate = AffineTransform(rotation = -self.pAngle)
        
        gKern =   NP.exp(-0.5*(xx**2 + yy**2))
        qSmoothSun = scipy.signal.fftconvolve(qSun,gKern) / dL**2
        qSmoothSun = qSmoothSun[dL//2:dL//2+self.sizeOfUv,dL//2:dL//2+self.sizeOfUv]
        smoothCoef = qSmoothSun[512, 512]
        qSmoothSun /= smoothCoef
        self.qSun_el_hd = warp(qSmoothSun,(shift + (rotate + back_shift)).inverse)
        
        res = warp(self.qSun_el_hd, (shift + (matrix + back_shift)).inverse)
        self.qSun_lm = warp(res,(shift + (scale + back_shift)).inverse)
        qSun_lm_fft = NP.fft.fft2(NP.roll(NP.roll(self.qSun_lm,self.sizeOfUv//2,0),self.sizeOfUv//2,1));
        qSun_lm_fft = NP.roll(NP.roll(qSun_lm_fft,self.sizeOfUv//2,0),self.sizeOfUv//2,1)# / self.sizeOfUv;
        qSun_lm_fft = NP.flip(qSun_lm_fft, 0)
#        qSun_lm_uv = qSun_lm_fft * uvPsf
#        qSun_lm_conv = NP.fft.fft2(NP.roll(NP.roll(qSun_lm_uv,self.sizeOfUv//2+1,0),self.sizeOfUv//2+1,1));
#        qSun_lm_conv = NP.roll(NP.roll(qSun_lm_conv,self.sizeOfUv//2-1,0),self.sizeOfUv//2-1,1);
#        qSun_lm_conv = NP.flip(NP.flip(qSun_lm_conv, 1), 0)
        self.lm_hd_relation[self.frequencyChannel] = NP.sum(self.qSun_lm)/NP.sum(self.qSun_el_hd)
        self.fftDisk = qSun_lm_fft #qSun_lm_conv, 
    
    def createUvUniform(self):
        self.uvUniform = NP.zeros((self.sizeOfUv, self.sizeOfUv), dtype = complex)
        flags_ew = NP.where(self.ewAntAmpLcp[self.frequencyChannel]==1e6)[0]
        flags_n = NP.where(self.nAntAmpLcp[self.frequencyChannel]==1e6)[0]
        O = self.sizeOfUv//2
        for i in range(self.antNumberN):
            for j in range(self.antNumberEW):
                if not (NP.any(flags_ew == j) or NP.any(flags_n == i)):
                    self.uvUniform[O + (i+1)*2, O + (j-32)*2] = 1
                    self.uvUniform[O - (i+1)*2, O - (j-32)*2] = 1
        for i in range(self.antNumberEW):
            if i != 32:
                if not (NP.any(flags_ew == i) or NP.any(flags_ew == 32)):
                    self.uvUniform[O, O + (i-32)*2] = 1
        self.uvUniform[O, O] = 1
        self.uvUniform /= NP.count_nonzero(self.uvUniform)
                    
    def createUvPsf(self, T, ewSlope, nSlope, shift):
        self.uvPsf = self.uvUniform.copy()
        O = self.sizeOfUv//2
        ewSlope = NP.deg2rad(ewSlope)
        nSlope = NP.deg2rad(nSlope)
        ewSlopeUv = NP.linspace(-O * ewSlope/2., O * ewSlope/2., self.sizeOfUv)
        nSlopeUv = NP.linspace(-O * nSlope/2., O * nSlope/2., self.sizeOfUv)
        ewGrid,nGrid = NP.meshgrid(ewSlopeUv, nSlopeUv)
        slopeGrid = ewGrid + nGrid
        slopeGrid[self.uvUniform == 0] = 0
        self.uvPsf *= T * NP.exp(1j * slopeGrid)
        self.uvPsf[O,O] = shift/NP.count_nonzero(self.uvUniform)
        
    def clipUv(self):
        O = self.sizeOfUv//2
        
        self.lcp_clipped = (self.lcp.real - 20000) * self.qSun_lm
        self.lcp_clipped = NP.flip(self.lcp_clipped, 0)
        self.lcp_clipped[self.lcp_clipped < 1e4] = 0
        lcp_clipped_fft = NP.fft.fft2(NP.roll(NP.roll(self.lcp_clipped,self.sizeOfUv//2,0),self.sizeOfUv//2,1));
        lcp_clipped_fft = NP.roll(NP.roll(lcp_clipped_fft,self.sizeOfUv//2,0),self.sizeOfUv//2,1)
        lcp_clipped_fft *= (3 * self.uvUniform * self.convolutionNormCoef/NP.count_nonzero(self.uvUniform))
        self.lcp_clipped_fft = lcp_clipped_fft
        self.uvLcp_qSun = self.uvLcp - self.lcp_clipped_fft
        
        self.shiftToAddLcp = self.lcpShift[self.frequencyChannel] - self.uvLcp_qSun[O,O] * NP.count_nonzero(self.uvUniform)
        # self.uvLcp_qSun[O,O] = 1/NP.count_nonzero(self.uvUniform)
        
        rcp_clipped = (self.rcp.real - 20000) * self.qSun_lm
        rcp_clipped = NP.flip(rcp_clipped, 0)
        rcp_clipped[rcp_clipped < 1e4] = 0
        rcp_clipped_fft = NP.fft.fft2(NP.roll(NP.roll(rcp_clipped,self.sizeOfUv//2,0),self.sizeOfUv//2,1));
        rcp_clipped_fft = NP.roll(NP.roll(rcp_clipped_fft,self.sizeOfUv//2,0),self.sizeOfUv//2,1)
        rcp_clipped_fft *= (3 * self.uvUniform * self.convolutionNormCoef/NP.count_nonzero(self.uvUniform))
        self.uvRcp_qSun = self.uvRcp - rcp_clipped_fft
        
        self.shiftToAddRcp = self.rcpShift[self.frequencyChannel] - self.uvRcp_qSun[O,O] * NP.count_nonzero(self.uvUniform)
        # self.uvRcp_qSun[O,O] = 1/NP.count_nonzero(self.uvUniform)
        
        # self.lcpShift[self.frequencyChannel] = 1
        # self.rcpShift[self.frequencyChannel] = 1
        
        
    def diskDiff(self, x, pol):
        self.createUvPsf(x[0], x[1], x[2], x[3])
        uvDisk = self.fftDisk * self.uvPsf
        if pol == 0:
            diff = self.uvLcp - uvDisk
        if pol == 1:
            diff = self.uvRcp - uvDisk
        return self.complex_to_real(diff[self.uvUniform!=0])
#        qSun_lm_conv = NP.fft.fft2(NP.roll(NP.roll(diff,uvSize//2+1,0),uvSize//2+1,1));
#        return NP.abs(NP.reshape(qSun_lm_conv, uvSize**2))

    def diskDiff_qSun(self, x, pol):
        self.createUvPsf(x[0], x[1], x[2], x[3])
        uvDisk = self.fftDisk * self.uvPsf
        if pol == 0:
            diff = self.uvLcp_qSun - uvDisk
        if pol == 1:
            diff = self.uvRcp_qSun - uvDisk
        return self.complex_to_real(diff[self.uvUniform!=0])
    
    def findDisk(self, clipped = False):
        Tb = self.ZirinQSunTb.getTbAtFrequency(self.freqList[self.frequencyChannel]*1e-6) * 1e3
        self.createDisk(sunpy.coordinates.sun.angular_radius(self.dateObs).to_value())
        self.createUvUniform()
        self.x_ini = [Tb/self.convolutionNormCoef,0,0,1]
        # x_ini = [1,0,0]
        if clipped:
            self.clipUv()
            self.center_ls_res_lcp = least_squares(self.diskDiff_qSun, self.x_ini, args = (0,))
            _diskLevelLcp, _ewSlopeLcp, _nSlopeLcp, _shiftLcp = self.center_ls_res_lcp['x']
            self.center_ls_res_rcp = least_squares(self.diskDiff_qSun, self.x_ini, args = (1,))
            _diskLevelRcp, _ewSlopeRcp, _nSlopeRcp, _shiftRcp = self.center_ls_res_rcp['x']
        else:
            self.center_ls_res_lcp = least_squares(self.diskDiff, self.x_ini, args = (0,))
            _diskLevelLcp, _ewSlopeLcp, _nSlopeLcp, _shiftLcp = self.center_ls_res_lcp['x']
            self.center_ls_res_rcp = least_squares(self.diskDiff, self.x_ini, args = (1,))
            _diskLevelRcp, _ewSlopeRcp, _nSlopeRcp, _shiftRcp = self.center_ls_res_rcp['x']
        
        # DISK LEVEL IS NEGATIVE???
        if _diskLevelLcp < 0 and _diskLevelRcp < 0:
            print('Disk optimization failed!')
            _diskLevelLcp = _diskLevelRcp = 1
        if _diskLevelLcp < 0 and _diskLevelRcp > 0:
            print('Disk optimization for LCP failed!')
            _diskLevelLcp = _diskLevelRcp
        if _diskLevelLcp > 0 and _diskLevelRcp < 0:
            print('Disk optimization for RCP failed!')
            _diskLevelRcp = _diskLevelLcp
        
        self.diskLevelLcp[self.frequencyChannel] = _diskLevelLcp
        self.diskLevelRcp[self.frequencyChannel] = _diskLevelRcp
        
        
        
        self.lcpShift[self.frequencyChannel] = self.lcpShift[self.frequencyChannel]/(_shiftLcp * self.convolutionNormCoef / Tb)
        self.rcpShift[self.frequencyChannel] = self.rcpShift[self.frequencyChannel]/(_shiftRcp * self.convolutionNormCoef / Tb)
        
        # if clipped:
        #     self.lcpShift[self.frequencyChannel]  += (self.shiftToAddLcp / (_diskLevelLcp*self.convolutionNormCoef / Tb))
        #     self.rcpShift[self.frequencyChannel]  += (self.shiftToAddRcp/ (_diskLevelRcp*self.convolutionNormCoef / Tb))
        # if not clipped:
        #     self.lcpShift[self.frequencyChannel] = self.lcpShift[self.frequencyChannel]/(_shiftLcp * self.convolutionNormCoef / Tb)
        #     self.rcpShift[self.frequencyChannel] = self.rcpShift[self.frequencyChannel]/(_shiftRcp * self.convolutionNormCoef / Tb)
        
        if not self.corr_amp_exist:
            self.ewAntAmpLcp[self.frequencyChannel][self.ewAntAmpLcp[self.frequencyChannel]!=1e6] *= NP.sqrt(_diskLevelLcp*self.convolutionNormCoef / Tb)
            self.nAntAmpLcp[self.frequencyChannel][self.nAntAmpLcp[self.frequencyChannel]!=1e6] *= NP.sqrt(_diskLevelLcp*self.convolutionNormCoef / Tb)
            self.ewAntAmpRcp[self.frequencyChannel][self.ewAntAmpRcp[self.frequencyChannel]!=1e6] *= NP.sqrt(_diskLevelRcp*self.convolutionNormCoef / Tb)
            self.nAntAmpRcp[self.frequencyChannel][self.nAntAmpRcp[self.frequencyChannel]!=1e6] *= NP.sqrt(_diskLevelRcp*self.convolutionNormCoef / Tb)
        
        self.ewSlopeLcp[self.frequencyChannel] = self.wrap(self.ewSlopeLcp[self.frequencyChannel] + _ewSlopeLcp)
        self.nSlopeLcp[self.frequencyChannel] = self.wrap(self.nSlopeLcp[self.frequencyChannel] + _nSlopeLcp)
        self.ewSlopeRcp[self.frequencyChannel] = self.wrap(self.ewSlopeRcp[self.frequencyChannel] + _ewSlopeRcp)
        self.nSlopeRcp[self.frequencyChannel] = self.wrap(self.nSlopeRcp[self.frequencyChannel] + _nSlopeRcp)
          
        
    def diskDiff_2(self, x, pol):
        self.createUvPsf(x[0], x[1], x[2])
        uvDisk = self.fftDisk * self.uvPsf
        if pol == 0:
            diff = self.uvLcp - uvDisk
        if pol == 1:
            diff = self.uvRcp - uvDisk
        return NP.sum(self.complex_to_real(diff[self.uvUniform!=0])**2)
#        qSun_lm_conv = NP.fft.fft2(NP.roll(NP.roll(diff,uvSize//2+1,0),uvSize//2+1,1));
#        return NP.abs(NP.reshape(qSun_lm_conv, uvSize**2))
    
    def findDisk_2(self):
        self.createDisk(980)
        self.createUvUniform()
        self.x_ini = [1,0,0]
        ls_res = basinhopping(self.diskDiff_2, self.x_ini, stepsize=180, minimizer_kwargs = {'args':(0,)})
        self.diskLevelLcp, self.ewSlopeLcp, self.nSlopeLcp = ls_res['x']
        ls_res = basinhopping(self.diskDiff_2, self.x_ini, stepsize=180, minimizer_kwargs = {'args':(1,)})
        self.diskLevelRcp, self.ewSlopeRcp, self.nSlopeRcp = ls_res['x']
        
    def findDisk_3(self):
        self.createDisk(980)
        self.createUvUniform()
        fun_lcp = 10
        fun_rcp = 10
        for i in range(3):
            for j in range(3):
                self.x_ini = [1, -90+i*90, -90+j*90]
                ls_res = least_squares(self.diskDiff, self.x_ini, args = (0,))
                print(NP.sum(ls_res['fun']**2))
                if NP.sum(ls_res['fun']**2)<fun_lcp:
                    print('min updated')
                    self.diskLevelLcp, self.ewSlopeLcp, self.nSlopeLcp = ls_res['x']
                    fun_lcp = NP.sum(ls_res['fun']**2)
                ls_res = least_squares(self.diskDiff, self.x_ini, args = (1,))
                if NP.sum(ls_res['fun']**2)<fun_rcp:
                    self.diskLevelRcp, self.ewSlopeRcp, self.nSlopeRcp = ls_res['x']
                    fun_rcp = NP.sum(ls_res['fun']**2)
        while self.ewSlopeLcp<-180:
            self.ewSlopeLcp+=360
        while self.ewSlopeLcp>180:
            self.ewSlopeLcp-=360
        while self.nSlopeLcp<-180:
            self.nSlopeLcp+=360
        while self.nSlopeLcp>180:
            self.nSlopeLcp-=360
        while self.ewSlopeRcp<-180:
            self.ewSlopeRcp+=360
        while self.ewSlopeRcp>180:
            self.ewSlopeRcp-=360
        while self.nSlopeRcp<-180:
            self.nSlopeRcp+=360
        while self.nSlopeRcp>180:
            self.nSlopeRcp-=360
        
    def findDisk_4(self):
        start_time_all = time.time()
        self.createDisk(980)
        self.createUvUniform()
        fun_lcp = 10
        fun_rcp = 10
        _diskLevelLcp = self.diskLevelLcp[self.frequencyChannel].copy()
        _diskLevelRcp = self.diskLevelRcp[self.frequencyChannel].copy()
                
        for i in range(3):
            for j in range(3):
                start_time = time.time()
                
                self.x_ini = [_diskLevelLcp, -90+i*90, -90+j*90]
                
                ls_res = least_squares(self.diskDiff, self.x_ini, args = (0,), ftol=self.centering_ftol)
                print(NP.sum(ls_res['fun']**2))
                if i==0 and j==0:
                    _diskLevelLcp, _ewSlopeLcp, _nSlopeLcp = ls_res['x']
                    fun_lcp = NP.sum(ls_res['fun']**2)
                    _diskLevelRcp, _ewSlopeRcp, _nSlopeRcp = ls_res['x']
                    fun_rcp = NP.sum(ls_res['fun']**2)
                    
                else:
                    if NP.sum(ls_res['fun']**2)<fun_lcp and ls_res['x'][0]>0:
                        print('min updated')
                        _diskLevelLcp, _ewSlopeLcp, _nSlopeLcp = ls_res['x']
                        fun_lcp = NP.sum(ls_res['fun']**2)
                        print((_diskLevelLcp, _ewSlopeLcp, _nSlopeLcp))
 
                    self.x_ini = [_diskLevelRcp, -90+i*90, -90+j*90]
                    ls_res = least_squares(self.diskDiff, self.x_ini, args = (1,), ftol=self.centering_ftol)
                    if NP.sum(ls_res['fun']**2)<fun_rcp and ls_res['x'][0]>0:
                        _diskLevelRcp, _ewSlopeRcp, _nSlopeRcp = ls_res['x']
                        fun_rcp = NP.sum(ls_res['fun']**2)
                    
                print("ITER " + str(i*3+j) + " --- %s seconds ---" % (time.time() - start_time))
                
        self.x_ini = [_diskLevelLcp, _ewSlopeLcp, _nSlopeLcp]               
        ls_res = least_squares(self.diskDiff, self.x_ini, args = (0,), ftol=1e-10)
        _diskLevelLcp, _ewSlopeLcp, _nSlopeLcp = ls_res['x']
        
        self.x_ini = [_diskLevelRcp, _ewSlopeRcp, _nSlopeRcp]               
        ls_res = least_squares(self.diskDiff, self.x_ini, args = (0,), ftol=1e-10)
        _diskLevelRcp, _ewSlopeRcp, _nSlopeRcp = ls_res['x']
        
        self.diskLevelLcp[self.frequencyChannel] = _diskLevelLcp
        self.diskLevelRcp[self.frequencyChannel] = _diskLevelRcp
        
        Tb = self.ZirinQSunTb.getTbAtFrequency(self.freqList[self.frequencyChannel]*1e-6) * 1e3
        
        self.ewAntAmpLcp[self.frequencyChannel][self.ewAntAmpLcp[self.frequencyChannel]!=1e6] *= NP.sqrt(_diskLevelLcp*self.convolutionNormCoef / Tb)
        self.nAntAmpLcp[self.frequencyChannel][self.nAntAmpLcp[self.frequencyChannel]!=1e6] *= NP.sqrt(_diskLevelLcp*self.convolutionNormCoef / Tb)
        self.ewAntAmpRcp[self.frequencyChannel][self.ewAntAmpRcp[self.frequencyChannel]!=1e6] *= NP.sqrt(_diskLevelRcp*self.convolutionNormCoef / Tb)
        self.nAntAmpRcp[self.frequencyChannel][self.nAntAmpRcp[self.frequencyChannel]!=1e6] *= NP.sqrt(_diskLevelRcp*self.convolutionNormCoef / Tb)
        
        self.ewSlopeLcp[self.frequencyChannel] = self.wrap(self.ewSlopeLcp[self.frequencyChannel] + _ewSlopeLcp)
        self.nSlopeLcp[self.frequencyChannel] = self.wrap(self.nSlopeLcp[self.frequencyChannel] + _nSlopeLcp)
        self.ewSlopeRcp[self.frequencyChannel] = self.wrap(self.ewSlopeRcp[self.frequencyChannel] + _ewSlopeRcp)
        self.nSlopeRcp[self.frequencyChannel] = self.wrap(self.nSlopeRcp[self.frequencyChannel] + _nSlopeRcp)
           
        print("FUNCTION RUNTIME  --- %s seconds ---" % (time.time() - start_time_all))
    
    def centerDisk(self):
        self.findDisk()
        self.buildEwPhase()
        self.buildNPhase()
        self.correctPhaseSlopeRL(self.frequencyChannel)
        self.buildEwPhase()
        self.buildNPhase()
        self.ewAntPhaLcp[self.frequencyChannel] += self.ewLcpPhaseCorrection[self.frequencyChannel]
        self.ewAntPhaRcp[self.frequencyChannel] += self.ewRcpPhaseCorrection[self.frequencyChannel]
        self.nAntPhaLcp[self.frequencyChannel] += self.nLcpPhaseCorrection[self.frequencyChannel]
        self.nAntPhaRcp[self.frequencyChannel] += self.nRcpPhaseCorrection[self.frequencyChannel]
        self.ewLcpPhaseCorrection[self.frequencyChannel] = NP.zeros(self.antNumberEW)
        self.ewRcpPhaseCorrection[self.frequencyChannel] = NP.zeros(self.antNumberEW)
        self.nLcpPhaseCorrection[self.frequencyChannel] = NP.zeros(self.antNumberN)
        self.nRcpPhaseCorrection[self.frequencyChannel] = NP.zeros(self.antNumberN)
        self.ewSlopeLcp[self.frequencyChannel] = 0
        self.ewSlopeRcp[self.frequencyChannel] = 0
        self.nSlopeLcp[self.frequencyChannel] = 0
        self.nSlopeRcp[self.frequencyChannel] = 0
        
    def modelDiskConv(self):
        # self.createUvPsf(self.diskLevelLcp,0,0,0)
        currentDiskTb = self.ZirinQSunTb.getTbAtFrequency(self.freqList[self.frequencyChannel]*1e-6)*1e3
        self.createUvPsf(currentDiskTb/self.convolutionNormCoef,0,0,currentDiskTb/self.convolutionNormCoef)
        self.uvDiskConv = self.fftDisk * self.uvPsf# - self.uvLcp
        qSun_lm = NP.fft.fft2(NP.roll(NP.roll(self.uvDiskConv,self.sizeOfUv//2+1,0),self.sizeOfUv//2+1,1));
        qSun_lm = NP.roll(NP.roll(qSun_lm,self.sizeOfUv//2-1,0),self.sizeOfUv//2-1,1)# / self.sizeOfUv;
        self.modelDisk = qSun_lm
        
    def modelDiskConv_unity(self):
        self.createDisk(980)
        self.createUvUniform()
        self.createUvPsf(1,0,0,1)
        self.uvDiskConv = self.fftDisk * self.uvPsf# - self.uvLcp
        qSun_lm = NP.fft.fft2(NP.roll(NP.roll(self.uvDiskConv,self.sizeOfUv//2+1,0),self.sizeOfUv//2+1,1));
        qSun_lm = NP.roll(NP.roll(qSun_lm,self.sizeOfUv//2-1,0),self.sizeOfUv//2-1,1)# / self.sizeOfUv;
        qSun_lm = NP.flip(qSun_lm, 0)
        self.modelDisk = qSun_lm