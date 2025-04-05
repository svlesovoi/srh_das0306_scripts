#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 17:27:13 2021

@author: maria
"""


from srhFitsFile36 import SrhFitsFile
import srh36MS2
import numpy as NP
import pylab as PL
import os, fnmatch
from astropy.io import fits
from skimage.transform import warp, AffineTransform
#from zeep import Client
import scipy.signal
import scipy.constants
import datetime
from ZirinTb import ZirinTb
from optparse import OptionParser
import skimage
#from scipy.stats import linregress
#import json
import ftplib;
import datetime as DT;
from casatasks import tclean, exportfits, flagdata
from casatools import image as IA
import json

gXSize = 1
gArcsecPerPixel = 1
gIImage = []
gVImage = []
gPSF = []
gFrequency = []
gDateObs = []
gImageHist = []
gEll_params = []

def generateEllipse(x0, y0, A, B, theta):
    N = 100
    t = NP.linspace(0,2*NP.pi,100)
    xy = NP.zeros((2,N))
    xy[0,:] = x0 + A*NP.cos(theta)*NP.cos(t) - B*NP.sin(theta)*NP.sin(t)
    xy[1,:] = y0 + A*NP.sin(theta)*NP.cos(t) + B*NP.cos(theta)*NP.sin(t)
    return xy
    
def fitFuncTb(f, A, B, C):
    return A + B*f + C*f**-1.8

def arcmin_format(xy, pos):
  return '%2d' % ((xy - gXSize/2) * gArcsecPerPixel / 60);

def findFits(path, pattern):
    result = [];
    for root, dirs, files in os.walk(path):
        for basename in files:
            if fnmatch.fnmatch(basename,pattern):
                if os.path.getsize(os.path.join(root,basename)) > 2880:
                    result.append(os.path.join(root,basename));
    return result;

def saveFitsImages(iImage, vImage, psfImage, saveCleanFitsPath, srhRawFits, frequencyIndex):
    resultArcsecPerPixel = 4.9
    srh_x_size = 512
    
    contours = (skimage.measure.find_contours(psfImage, 0.5))[0]
    con = NP.zeros_like(contours)
    con[:,1] = contours[:,0]
    con[:,0] = contours[:,1]
    gPSF.append(psfImage)
    sunEll = skimage.measure.EllipseModel()
    ellipseEstimateResult = sunEll.estimate(con)
    
    
    pHeader = fits.Header();
    pHeader['DATE-OBS']     = srhRawFits.hduList[0].header['DATE-OBS']
    pHeader['T-OBS']        = srhRawFits.hduList[0].header['TIME-OBS']
    pHeader['INSTRUME']     = srhRawFits.hduList[0].header['INSTRUME']
    pHeader['ORIGIN']       = srhRawFits.hduList[0].header['ORIGIN']
    pHeader['FREQUENC']     = ('%d') % (srhRawFits.freqList[frequencyIndex]/1e3 + 0.5)
    pHeader['CDELT1']       = resultArcsecPerPixel
    pHeader['CDELT2']       = resultArcsecPerPixel
    pHeader['CRPIX1']       = srh_x_size // 2
    pHeader['CRPIX2']       = srh_x_size // 2
    pHeader['CTYPE1']       = 'HPLN-TAN'
    pHeader['CTYPE2']       = 'HPLT-TAN'
    pHeader['CUNIT1']       = 'arcsec'
    pHeader['CUNIT2']       = 'arcsec'
    if (ellipseEstimateResult):
        pHeader['PSF_ELLA']     = NP.ceil(sunEll.params[2] * resultArcsecPerPixel*10)/10 # PSF ellipse A arcsec
        pHeader['PSF_ELLB']     = NP.ceil(sunEll.params[3] * resultArcsecPerPixel*10)/10 # PSF ellipse B arcsec
        pHeader['PSF_ELLT']     = sunEll.params[4] # PSF ellipse theta rad

    saveFitsIhdu = fits.PrimaryHDU(header=pHeader, data=iImage.astype('float32'))
    saveFitsIpath = saveCleanFitsPath + 'srh_I_%sT%s_%04d.fit'%(srhRawFits.hduList[0].header['DATE-OBS'], srhRawFits.hduList[0].header['TIME-OBS'], srhRawFits.freqList[frequencyIndex]*1e-3 + .5)
    ewLcpPhaseColumn = fits.Column(name='ewLcpPhase', format='D', array = srhRawFits.ewAntPhaLcp[frequencyIndex,:] + srhRawFits.ewLcpPhaseCorrection[frequencyIndex,:])
    ewRcpPhaseColumn = fits.Column(name='ewRcpPhase', format='D', array = srhRawFits.ewAntPhaRcp[frequencyIndex,:] + srhRawFits.ewRcpPhaseCorrection[frequencyIndex,:])
    sLcpPhaseColumn = fits.Column(name='nLcpPhase',   format='D', array = srhRawFits.nAntPhaLcp[frequencyIndex,:] + srhRawFits.nLcpPhaseCorrection[frequencyIndex,:])
    sRcpPhaseColumn = fits.Column(name='nRcpPhase',   format='D', array = srhRawFits.nAntPhaRcp[frequencyIndex,:] + srhRawFits.nRcpPhaseCorrection[frequencyIndex,:])
    saveFitsIExtHdu = fits.BinTableHDU.from_columns([ewLcpPhaseColumn, ewRcpPhaseColumn, sLcpPhaseColumn, sRcpPhaseColumn])
    hduList = fits.HDUList([saveFitsIhdu, saveFitsIExtHdu])
    hduList.writeto(saveFitsIpath)
    
    saveFitsVhdu = fits.PrimaryHDU(header=pHeader, data=vImage.astype('float32'))
    saveFitsVpath = saveCleanFitsPath + 'srh_V_%sT%s_%04d.fit'%(srhRawFits.hduList[0].header['DATE-OBS'], srhRawFits.hduList[0].header['TIME-OBS'], srhRawFits.freqList[frequencyIndex]*1e-3 + .5)
    hduList = fits.HDUList(saveFitsVhdu)
    hduList.writeto(saveFitsVpath)

currentDate = datetime.datetime.now().date().strftime("%Y%m%d")
parser = OptionParser()
#parser.add_option("-f", "--file", dest="fitPath", default = '/home/sergey_lesovoi/SRH_DATA/SRH/SRH0306/' + currentDate + '/')
parser.add_option("-f", "--file", dest="fitPath", default = '/home/svlesovoi/SRH_0306/' + currentDate + '/')
parser.add_option("-t", "--treshold", dest="cleanTresh", default = '0.1mJy')
parser.add_option("-s", "--scan", dest="scan", default = '0~19')
#cleanPath = '/home/sergey_lesovoi/SRH_DATA/SRH/SRH0306/cleanMaps/'
cleanPath = '/home/svlesovoi/SRH_0306/cleanMaps/'

(clean_options, clean_args) = parser.parse_args()
fitPath = clean_options.fitPath
cleanTresh = clean_options.cleanTresh
scan = clean_options.scan

fitNames = findFits(fitPath,'*.fit')
fitNames.sort()
fileName = fitNames[-1]

ZirinQSunTb = ZirinTb()

file = SrhFitsFile(fileName, 1025)
file.useNonlinearApproach = True
file.getHourAngle(0)
#freqList = [2,7,13]
freqList = [1]
frequencyNumber = len(freqList)
gIImage = NP.zeros((frequencyNumber, 512, 512))
gVImage = NP.zeros((frequencyNumber, 512, 512))

pAngle = file.pAngle

#pb_level = [0.97, 0.96, 0.95]
pb_level = [0.94, 0.89, 0.82]

ia = IA()

if (not os.path.exists(cleanPath + currentDate)):
    os.system('mkdir ' + cleanPath + currentDate)
    
for frequency in range(frequencyNumber):
    freq_mhz = int(file.freqList[freqList[frequency]]/1e3)
    saveName = 'MS/' + fileName.split('/')[-1].split('.')[0] + '_' + str(freq_mhz) + '.ms'
    file.calibrate(freqList[frequency], average = 20)
    ms2Table = srh36MS2.Srh36Ms2(saveName)
    ms2Table.createMS(file, frequencyChannel = [int(freqList[frequency])], phaseCorrect = True, amplitudeCorrect = True)
        
#    flagdata(vis = saveName, antenna = '1,2,3,4')
    
    imName = 'images/' + str(freqList[frequency]) + '_0306'
    tclean(vis = saveName,
           imagename = imName,
           niter = 10000,
           threshold = '0.12Jy',
           cell = 2.45,
           imsize = [1024,1024],
           stokes = 'RRLL',
           usemask = 'pb',
           pbmask = pb_level[frequency],
           scan = '0~19')
    
    ia.open(imName + '.image')
    rcp = ia.getchunk()[:,:,0,0].transpose()
    lcp = ia.getchunk()[:,:,1,0].transpose()
    ia.close()
    
    lam = scipy.constants.c/(file.freqList[freqList[frequency]]*1e3)
    lcp = lcp * lam**2 * 1e-22 / (2*scipy.constants.k * file.beam_sr[freqList[frequency]])
    rcp = rcp * lam**2 * 1e-22 / (2*scipy.constants.k * file.beam_sr[freqList[frequency]])
    
    rcp += (file.rcpShift[freqList[frequency]]/NP.count_nonzero(file.uvLcp))
    lcp += (file.lcpShift[freqList[frequency]]/NP.count_nonzero(file.uvRcp))
    
    ia.open(imName + '.psf')
    psf = ia.getchunk()[:,:,0,0].transpose()
    ia.close()
    
    clnjunks = ['.flux', '.mask', '.model', '.psf', '.residual','.sumwt','.pb','.image']
    for clnjunk in clnjunks:
        if os.path.exists(imName + clnjunk):
            os.system('rm -rf '+imName + clnjunk)
      
    srh_y_size = 1024
    srh_x_size = 1024
    O = srh_x_size//2
    Q = srh_x_size//4
    scale = AffineTransform(scale=(0.5,0.5))
    shift = AffineTransform(translation=(-srh_y_size/2,-srh_y_size/2))
    rotate = AffineTransform(rotation = -pAngle)
    back_shift = AffineTransform(translation=(srh_y_size/2,srh_y_size/2))
    
    rcp = warp(rcp,(shift + (rotate + back_shift)).inverse)
    lcp = warp(lcp,(shift + (rotate + back_shift)).inverse)
    psf = warp(psf,(shift + (rotate + back_shift)).inverse)
    rcp = warp(rcp,(shift + (scale + back_shift)).inverse)[O-Q:O+Q,O-Q:O+Q]
    lcp = warp(lcp,(shift + (scale + back_shift)).inverse)[O-Q:O+Q,O-Q:O+Q]
    psf = warp(psf,(shift + (scale + back_shift)).inverse)[O-Q:O+Q,O-Q:O+Q]
        
    saveFitsImages((rcp + lcp)/2, (rcp - lcp)/2, psf, cleanPath + currentDate + '/', file, freqList[frequency])
    
    gIImage[frequency] = (rcp + lcp)/2
    gVImage[frequency] = (rcp - lcp)/2
    gFrequency.append(freq_mhz)
    
    clnjunks = ['.ms', '.ms.flagversions', '*clean*.fit']
    for clnjunk in clnjunks:
        if os.path.exists(saveName.split('.')[0] + clnjunk):
            os.system('rm -rf ' + saveName.split('.')[0] + clnjunk)
    os.system('rm -rf *casa*.log')
    
    

#fig, pl = PL.subplots(nrows=frequencyNumber,ncols=2,figsize=(5.6,7.9))
#fig.subplots_adjust(hspace=0.001,wspace=0.001,top=0.95,bottom=0.01,left=0.01,right=0.99)
#TI = 2e5
#TV = 2e4
#imageDate = file.hduList[0].header['DATE-OBS'].replace('-','')
#imageTimeShow = file.hduList[0].header['TIME-OBS'][0:5]
#imageTime = file.hduList[0].header['TIME-OBS'].replace(':','')[0:4]
#combinedImagePath = 'srh_%s_%s.png'%(imageDate,imageTime)
#fig.suptitle('%s, %s UT'%(imageDate, imageTimeShow))
#for row in range(frequencyNumber):
#    pl[row,0].axis('off')
#    pl[row,0].imshow(gIImage[row],cmap='hot',vmax=TI,vmin=0,origin='lower')
#    flux = 2*2*gIImage[row][256-200:256+200,256-200:256+200].sum()*scipy.constants.k/(scipy.constants.c/gFrequency[row]*1e-6)**2 * NP.deg2rad(4.911/3600)**2 / 1e-22
#    # hist,bins = NP.histogram(gIImage[row],bins=50,range=(-2e4,-1e3))
#    # maxInd = NP.argmax(hist)
#    # skyLevel = bins[maxInd]
#    flux = 1.5*2*(gIImage[row]).sum()*scipy.constants.k/(scipy.constants.c/gFrequency[row]*1e-6)**2 * NP.deg2rad(4.911/3600)**2 / 1e-22
#    levels = NP.linspace(TI,NP.max(gIImage[row]),3)
#    pl[row,0].contour(gIImage[row],cmap='hot',levels=levels,linewidths=0.5,origin='lower')
#    pl[row,0].text(10,10,'I %.1f s.f.u.'%flux,color='white')
    # flux = 2*gIImage[row][256-200:256+200,256-200:256+200].sum()*scipy.constants.k/(scipy.constants.c/gFrequency[row]*1e-6)**2 * NP.deg2rad(gArcsecPerPixel/3600)**2 / 1e-22
    # pl[row,0].text(10,gXSize - 30,'F = %.1f sfu'%flux,color='white')

    # ellXY = generateEllipse(50,50,gEll_params[row][2],gEll_params[row][3],gEll_params[row][4])
    # pl[row,0].plot(ellXY[0,:],ellXY[1,:],color='white')

#    pl[row,1].axis('off')
#    pl[row,1].imshow(gVImage[row],cmap='gray',vmax=TV,vmin=-TV,origin='lower')
#    levels = NP.linspace(TV,NP.max(gVImage[row]),3)
#    pl[row,1].contour(gVImage[row],cmap='gray',levels=levels,linewidths=0.5,origin='lower')
#    levels = NP.linspace(NP.min(gVImage[row]),-TV,3)
#    pl[row,1].contour(gVImage[row],cmap='gray',levels=levels,linewidths=0.5,origin='lower')
#    pl[row,1].text(10,512 - 30,'%d MHz'%gFrequency[row],color='white')
#    pl[row,1].text(10,10,'V',color='white')

# fig.savefig(combinedImagePath)
# fdSrhImageName = open('srhImageName.txt','w');
# fdSrhImageName.write(combinedImagePath);
# fdSrhImageName.close();


# dateName = DT.datetime.now().strftime("%Y%m%d")
# fd = ftplib.FTP('10.1.1.9','sergey','jill21');

# fi = open('srhImageName.txt','rb');
# fd.storbinary('STOR /Public/sergey/corrPlots/srhImageName.txt',fi);
# fi.close();

# fi = open('srhImageName.txt');
# cleanImageName = fi.readline()
# fCleanImage = open(cleanImageName,'rb')
# fd.storbinary('STOR /Public/sergey/corrPlots/' + cleanImageName,fCleanImage);
# fCleanImage.close()
# fi.close()
# fd.quit();


# saving visibilities to json file for uv-plane telemetry
frequencyChannel = 0
visLcp = NP.mean(NP.abs(file.visLcp[frequencyChannel,:,:3007]), 0)
visRcp = NP.mean(NP.abs(file.visRcp[frequencyChannel,:,:3007]), 0)

t_shape=(31, 97)

visLcp_r = NP.reshape(visLcp*0.5, t_shape)
visRcp_r = NP.reshape(visRcp*0.5, t_shape)

Lcp = NP.round(visLcp_r, 2)
Rcp = NP.round(visRcp_r, 2)

data = {"last_update_0306": str(datetime.datetime.utcnow()),\
        "grid0306": {"lcp": Lcp.tolist(), "rcp": Rcp.tolist()}}

json.dump(data, open("uvplane0306.json", "w"))

import requests

url = "http://10.1.2.40/api/uvplane/add"
array = "SRH0306"
token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VybmFtZSI6Imdsb2JhIiwiZXhwIjowLCJ2ZXJzaW9uIjowfQ.RnCxNrzzMajETRZkhntvXu2-0NOolp4taEm0-hEbk6Y'

response = requests.post(url, json={"password": token, array: data})

if response.status_code == 201:
    print("Success:", response.json())
else:
    print("Error:", response.content)

