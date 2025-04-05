#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 06:32:14 2021

@author: svlesovoi
"""

import datetime as DT;
from astropy.io import fits
import srh0612Utils
import numpy as NP
import pylab as PL
from srhFitsFile36 import SrhFitsFile
import matplotlib
import ftplib;
from ZirinTb import ZirinTb

ant_idx = NP.linspace(0,127,128,dtype='int')
not_ant_idx = [41,47,56,59,60,97,124]
ant_idx = list(set(ant_idx) - set(not_ant_idx))

def hhmm_format(t, pos):
  hh = (int)(t / 3600.);
  t -= hh*3600.;
  mm = (int)(t / 60.);
  return '%02d:%02d' % (hh,mm);

dateName = DT.datetime.now().strftime("%Y%m%d")
try:
    previousCpFits = fits.open('srh_0306_cp_' + dateName + '.fits')
except FileNotFoundError:
    previousCpFits = 0

dt_major = 3600.;
dt_minor = 900.;

currentRawFitses = srh0612Utils.findFits('SRH0306/' + dateName ,'*.fit')  
currentRawFitses.sort()
currentRawFitses = NP.array(currentRawFitses[1:])

Zi = ZirinTb()

if (previousCpFits):
    previousRawFitses = previousCpFits[3].data['rawFitsNames']
    fitNames = currentRawFitses[previousRawFitses.shape[0]:]
else:
    fitNames = currentRawFitses

#for fName in fitNames:
#    print(fName)
#    sF = SrhFitsFile(fName, 256)
#    zerInd = NP.where(sF.ampLcp_c <= 0.01)
#    sF.ampLcp_c[zerInd] = 1e6
#    zerInd = NP.where(sF.ampRcp_c <= 0.01)
#    sF.ampRcp_c[zerInd] = 1e6

firstVis = 0
lastVis = 3007
minAmp = 0.01
for fName in fitNames:
    print(fName)
    sF = SrhFitsFile(fName, 512, flux_norm = False)
#    zerInd = NP.where(sF.ampLcp_c < minAmp)
#    sF.ampLcp_c[zerInd] = 1e6
#    zerInd = NP.where(sF.ampRcp_c < minAmp)
#    sF.ampRcp_c[zerInd] = 1e6
#    for vis in NP.arange(firstVis,lastVis):
#        AB = sF.visIndex2antIndex(vis)
#        indA = NP.where(sF.antennaNumbers == str(AB[0]))[0][0]
#        indB = NP.where(sF.antennaNumbers == str(AB[1]))[0][0]
#        sF.visRcp[:,:,vis] = sF.visRcp[:,:,vis] / ((NP.sqrt(sF.ampRcp_c[:,:,indA] * sF.ampRcp_c[:,:,indB])))
#        sF.visLcp[:,:,vis] = sF.visLcp[:,:,vis] / ((NP.sqrt(sF.ampLcp_c[:,:,indA] * sF.ampLcp_c[:,:,indB])))
    if (fName == fitNames[0]):
        freqAmount  = sF.freqListLength;
        freqList = sF.freqList;
        startTime= DT.datetime.strptime(sF.hduList[0].header['DATE-OBS'] + ' ' + sF.hduList[0].header['TIME-OBS'], '%Y-%m-%d %H:%M:%S');
        times = sF.freqTime;
        corrFluxRcp = NP.mean(NP.abs(sF.visRcp[:,:,firstVis:lastVis]),axis=2);
        corrFluxLcp = NP.mean(NP.abs(sF.visLcp[:,:,firstVis:lastVis]),axis=2);
        ampFluxRcp = NP.mean(sF.ampRcp[:,:,ant_idx], axis = 2)
        ampFluxLcp = NP.mean(sF.ampLcp[:,:,ant_idx], axis = 2)
        try:
            validPacketNumber = sF.correctSubpacketsNumber
            subpacketLcpIdx = NP.where(sF.subpacketLcp != validPacketNumber)
            subpacketRcpIdx = NP.where(sF.subpacketRcp != validPacketNumber)
            # print(subpacketLcpIdx)
            corrFluxLcp[subpacketLcpIdx] = 0.
            corrFluxRcp[subpacketRcpIdx] = 0.
            corrFluxLcp[subpacketRcpIdx] = 0.
            corrFluxRcp[subpacketLcpIdx] = 0.
            ampFluxLcp[subpacketLcpIdx] = 0.
            ampFluxRcp[subpacketRcpIdx] = 0.
            ampFluxLcp[subpacketRcpIdx] = 0.
            ampFluxRcp[subpacketLcpIdx] = 0.
        except:
            pass
    else:
        times = NP.concatenate((times,sF.freqTime),axis=1);
        corrRcp_temp = NP.mean(NP.abs(sF.visRcp[:,:,firstVis:lastVis]),axis=2)
        corrLcp_temp = NP.mean(NP.abs(sF.visLcp[:,:,firstVis:lastVis]),axis=2)
        ampRcp_temp = NP.mean(sF.ampRcp[:,:,ant_idx], axis = 2)
        ampLcp_temp = NP.mean(sF.ampLcp[:,:,ant_idx], axis = 2)
        try:
            validPacketNumber = sF.correctSubpacketsNumber
            subpacketLcpIdx = NP.where(sF.subpacketLcp != validPacketNumber)
            subpacketRcpIdx = NP.where(sF.subpacketRcp != validPacketNumber)
            # print(subpacketLcpIdx)
            corrLcp_temp[subpacketLcpIdx] = 0.
            corrRcp_temp[subpacketRcpIdx] = 0.
            corrLcp_temp[subpacketRcpIdx] = 0.
            corrRcp_temp[subpacketLcpIdx] = 0.
            ampLcp_temp[subpacketLcpIdx] = 0.
            ampRcp_temp[subpacketRcpIdx] = 0.
            ampLcp_temp[subpacketRcpIdx] = 0.
            ampRcp_temp[subpacketLcpIdx] = 0.
        except:
            pass
        
        corrFluxRcp = NP.concatenate((corrFluxRcp, corrRcp_temp),axis=1);
        corrFluxLcp = NP.concatenate((corrFluxLcp, corrLcp_temp),axis=1);
        ampFluxRcp = NP.concatenate((ampFluxRcp, ampRcp_temp), axis = 1)
        ampFluxLcp = NP.concatenate((ampFluxLcp, ampLcp_temp), axis = 1)
        
    sF.close();

#    for vis in range(3007):
#        AB = sF.visIndex2antIndex(vis)
#        indA = NP.where(sF.antennaNumbers == str(AB[0]))[0][0]
#        indB = NP.where(sF.antennaNumbers == str(AB[1]))[0][0]
#        sF.visLcp[:,:,vis] = sF.visLcp[:,:,vis] / ((NP.sqrt(sF.ampLcp_c[:,:,indA] * sF.ampLcp_c[:,:,indB])))
#        sF.visRcp[:,:,vis] = sF.visRcp[:,:,vis] / ((NP.sqrt(sF.ampRcp_c[:,:,indA] * sF.ampRcp_c[:,:,indB])))
#    if (fName == fitNames[0]):
#        freqAmount  = sF.freqListLength;
#        freqList = sF.freqList;
#        startTime= DT.datetime.strptime(sF.hduList[0].header['DATE-OBS'] + ' ' + sF.hduList[0].header['TIME-OBS'], '%Y-%m-%d %H:%M:%S');
#        times = sF.freqTime;
#        r_real = abs(sF.visRcp.real[:,:,0:3007])
#        r_imag = abs(sF.visRcp.imag[:,:,0:3007])
#        corrFluxRcp = NP.mean(NP.sqrt(r_real**2. + r_imag**2.),axis=2);
#        r_real = abs(sF.visLcp.real[:,:,0:3007])
#        r_imag = abs(sF.visLcp.imag[:,:,0:3007])
#        corrFluxLcp = NP.mean(NP.sqrt(r_real**2. + r_imag**2.),axis=2);
#        r_real = 0;
#        r_imag = 0;
#        ampFluxRcp = NP.mean(sF.ampRcp, axis = 2)
#        ampFluxLcp = NP.mean(sF.ampLcp, axis = 2)
#    else:
#        times = NP.concatenate((times,sF.freqTime),axis=1);
#        r_real = abs(sF.visRcp.real[:,:,0:3007])
#        r_imag = abs(sF.visRcp.imag[:,:,0:3007])
#        corrFluxRcp = NP.concatenate((corrFluxRcp, NP.mean(NP.sqrt(r_real**2. + r_imag**2.),axis=2)),axis=1);
#        r_real = abs(sF.visLcp.real[:,:,0:3007])
#        r_imag = abs(sF.visLcp.imag[:,:,0:3007])
#        corrFluxLcp = NP.concatenate((corrFluxLcp, NP.mean(NP.sqrt(r_real**2. + r_imag**2.),axis=2)),axis=1);
#        r_real = 0;
#        r_imag = 0;
#        ampFluxRcp = NP.concatenate((ampFluxRcp, NP.mean(sF.ampRcp, axis = 2)), axis = 1)
#        ampFluxLcp = NP.concatenate((ampFluxLcp, NP.mean(sF.ampLcp, axis = 2)), axis = 1)
        
#    sF.close();

# vNorm = NP.array([1.0152989 , 1.00538075, 1.00851714, 1.00820023, 1.00913836,\
#        1.01499069, 1.02098195, 1.02310024, 1.01732735, 1.0143846, \
#        1.00318449, 1.00069164, 1.00531617, 1.01528238, 1.02474869, \
#        1.0279521 ])

saveFreqs = NP.linspace(0,freqAmount-1,freqAmount,dtype=NP.uint8)
saveTime = NP.zeros((saveFreqs.shape[0],times.shape[1]))
saveCorrI = NP.zeros((saveFreqs.shape[0],times.shape[1]))
saveCorrV = NP.zeros((saveFreqs.shape[0],times.shape[1]))
saveFluxI = NP.zeros((saveFreqs.shape[0],ampFluxRcp.shape[1]))
saveFluxV = NP.zeros((saveFreqs.shape[0],ampFluxRcp.shape[1]))
saveFluxRcp = NP.zeros((saveFreqs.shape[0],ampFluxRcp.shape[1]))
saveFluxLcp = NP.zeros((saveFreqs.shape[0],ampFluxRcp.shape[1]))

#for ff in saveFreqs:
#    ampFluxRcp[ff,:] *= vNorm[ff]


for ff in saveFreqs:
    saveTime[ff,:] = times[ff,:]
    saveCorrI[ff,:] = 0.5*(corrFluxRcp[ff,:] + corrFluxLcp[ff,:])
    saveCorrV[ff,:] = 0.5*(corrFluxRcp[ff,:] - corrFluxLcp[ff,:])
    saveFluxI[ff,:] = ampFluxRcp[ff,:] + ampFluxLcp[ff,:]
    saveFluxV[ff,:] = ampFluxRcp[ff,:] - ampFluxLcp[ff,:]
    saveFluxRcp[ff,:] = ampFluxRcp[ff,:]
    saveFluxLcp[ff,:] = ampFluxLcp[ff,:]

try:
    previousCpZerosFits = fits.open('srh_0306_cp_zeros.fits')
    corrZeros = previousCpZerosFits[2].data['corrI']
    fluxZeros = previousCpZerosFits[2].data['fluxI']
    fluxZerosLcp = previousCpZerosFits[2].data['skyLcp']
    fluxZerosRcp = previousCpZerosFits[2].data['skyRcp']
except FileNotFoundError:
    corrZeros = NP.zeros_like(freqList)
    fluxZeros = NP.zeros_like(freqList)
    fluxZerosLcp = NP.zeros_like(freqList)
    fluxZerosRcp = NP.zeros_like(freqList)

try:
    previousCpFluxNormFits = fits.open('srh_0306_cp_fluxNorm.fits')
    fluxNormI = previousCpFluxNormFits[2].data['fluxNormI']
    fluxNormLcp = previousCpFluxNormFits[2].data['fluxNormLcp']
    fluxNormRcp = previousCpFluxNormFits[2].data['fluxNormRcp']
except FileNotFoundError:
    fluxNormI = NP.ones_like(freqList)*0.01
    fluxNormRcp = NP.ones_like(freqList)*0.01
    fluxNormLcp = NP.ones_like(freqList)*0.01

for ff in saveFreqs:
    # saveCorrI[ff,:] -= corrZeros[ff]
    # saveFluxI[ff,:] -= fluxZeros[ff]
    saveCorrI[ff][saveCorrI[ff]!=0] -= corrZeros[ff]
    saveFluxI[ff][saveFluxI[ff]!=0] -= fluxZeros[ff]
    saveFluxI[ff,:] *= fluxNormI[ff]
    saveFluxV[ff,:] *= fluxNormI[ff]
    saveFluxRcp[ff,:] -= fluxZerosRcp[ff]
    saveFluxLcp[ff,:] -= fluxZerosLcp[ff]
    saveFluxRcp[ff,:] *= fluxNormRcp[ff]
    saveFluxLcp[ff,:] *= fluxNormLcp[ff]

if (previousCpFits):
    previousTime = previousCpFits[2].data['time']
    previousCorrI = previousCpFits[2].data['I']
    previousCorrV = previousCpFits[2].data['V']
    previousFluxI = previousCpFits[2].data['flux_I']
    previousFluxV = previousCpFits[2].data['flux_V']
    previousFluxRcp = previousCpFits[2].data['flux_RCP']
    previousFluxLcp = previousCpFits[2].data['flux_LCP']
    saveTime = NP.concatenate((previousTime, saveTime), axis=1)
    saveCorrI = NP.concatenate((previousCorrI, saveCorrI),axis=1)
    saveCorrV = NP.concatenate((previousCorrV, saveCorrV),axis=1)
    saveFluxI = NP.concatenate((previousFluxI, saveFluxI),axis=1)
    saveFluxV = NP.concatenate((previousFluxV, saveFluxV),axis=1)
    saveFluxRcp = NP.concatenate((previousFluxRcp, saveFluxRcp),axis=1)
    saveFluxLcp = NP.concatenate((previousFluxLcp, saveFluxLcp),axis=1)
    
# creating corrPlot
c_list = matplotlib.colors.LinearSegmentedColormap.from_list(PL.cm.datad['gist_rainbow'], colors=['r','g','b'], N = freqAmount)
fig = PL.figure(figsize = (16,8));
fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05)
fig.tight_layout()
commonTitle = 'Correlation plot ' + startTime.strftime('%Y %B %d') + '. Stokes I, V at 3-6 GHz'
fig.suptitle(commonTitle,fontsize=14)

sub = fig.add_subplot(1,1,1)
sub.set_ylabel('correlation coefficient')
sub.set_xlabel('UT')
sub.xaxis.set_major_locator(PL.MultipleLocator(dt_major))
sub.xaxis.set_major_formatter(PL.FuncFormatter(hhmm_format))
sub.xaxis.set_minor_locator(PL.MultipleLocator(dt_minor))
sub.set_xlim((0.*3600.,10.*3600.))
sub.set_ylim((-0.01,0.4))
for ff in saveFreqs:
    sub.scatter(saveTime[ff,:],saveCorrI[ff,:],color=c_list(ff), s=0.5, linewidths = 0,label='%d MHz'%(freqList[ff]*1e-3))
    sub.scatter(saveTime[ff,:],saveCorrV[ff,:],color=c_list(ff), s=0.5, linewidths = 0)
    sub.legend(markerscale=10)
corrPlotName = 'fCorrPlot'+ dateName
corrPlotName_png = corrPlotName + '.png'
PL.savefig(corrPlotName_png)

# creating fluxPlot
fig = PL.figure(figsize = (16,8))
fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05)
fig.tight_layout()
commonTitle = 'Flux plot ' + startTime.strftime('%Y %B %d') + '. Stokes I, V at 3-6 GHz'
fig.suptitle(commonTitle,fontsize=14)

sub = fig.add_subplot(1,1,1)
sub.set_ylabel('s.f.u.')
sub.set_xlabel('UT')
sub.xaxis.set_major_locator(PL.MultipleLocator(dt_major))
sub.xaxis.set_major_formatter(PL.FuncFormatter(hhmm_format))
sub.xaxis.set_minor_locator(PL.MultipleLocator(dt_minor))
sub.set_xlim((0.*3600.,10.*3600.))
sub.set_ylim((-2e1, 1e3))

for ff in saveFreqs:
    # sub.scatter(saveTime[ff,:],saveFluxI[ff,:],color=c_list(ff), s=1.5, linewidths = 0,label='%d MHz'%(freqList[ff]*1e-3))
    # sub.scatter(saveTime[ff,:],saveFluxV[ff,:],color=c_list(ff), s=1.5, linewidths = 0)
    sub.scatter(saveTime[ff,:],saveFluxRcp[ff,:] + saveFluxLcp[ff,:],color=c_list(ff), s=1.5, linewidths = 0,label='%d MHz'%(freqList[ff]*1e-3))
    sub.scatter(saveTime[ff,:],saveFluxRcp[ff,:] - saveFluxLcp[ff,:],color=c_list(ff), s=1.5, linewidths = 0)
    sub.legend(markerscale=5)
fluxPlotName = 'fFluxPlot'+ dateName
fluxPlotName_png = fluxPlotName + '.png'
PL.savefig(fluxPlotName_png)

# writing srh_cp file
    
dataFormat = str(saveTime.shape[1]) + 'D'
freqsColumn = fits.Column(name='frequencies',format='D',array=freqList[saveFreqs])
timeColumn = fits.Column(name='time',format=dataFormat,array=saveTime)
IColumn = fits.Column(name='I',format=dataFormat,array=saveCorrI)
VColumn = fits.Column(name='V',format=dataFormat,array=saveCorrV)
IAmpColumn = fits.Column(name='flux_I',format=dataFormat,array=saveFluxI)
VAmpColumn = fits.Column(name='flux_V',format=dataFormat,array=saveFluxV)
RcpAmpColumn = fits.Column(name='flux_RCP',format=dataFormat,array=saveFluxRcp)
LcpAmpColumn = fits.Column(name='flux_LCP',format=dataFormat,array=saveFluxLcp)

fTableHdu = fits.BinTableHDU.from_columns([freqsColumn]);
dTableHdu = fits.BinTableHDU.from_columns([timeColumn, IColumn, VColumn, IAmpColumn, VAmpColumn, RcpAmpColumn, LcpAmpColumn])
pHeader = fits.Header()
pHeader['DATE-OBS']     = sF.hduList[0].header['DATE-OBS']
pHeader['TIME-OBS']     = sF.hduList[0].header['TIME-OBS']
pHeader['INSTRUME']     = 'SRH'
pHeader['ORIGIN']       = 'ISTP'
pHeader['OBS-LAT']      = '51.759'
pHeader['OBS-LONG']     = '102.217'
pHeader['OBS-ALT']      = '799'
pHeader['FR_CHAN']      = '10'

fitsNamesColumn = fits.Column(name='rawFitsNames', format='A256', array=currentRawFitses)
aTableHdu = fits.TableHDU.from_columns([fitsNamesColumn])

pHdu = fits.PrimaryHDU(header=pHeader)
hduList = fits.HDUList([pHdu, fTableHdu, dTableHdu, aTableHdu])
hduList.writeto('srh_0306_cp_' + dateName + '.fits',overwrite=True)
hduList.writeto('SRH0306/corrPlots/srh_0306_cp_' + dateName + '.fits',overwrite=True)
hduList.close()

#zeroCorrIColumn = fits.Column(name='corrI',format='D',array=saveCorrI[:,-70:].mean(axis=1))
#zeroFluxIColumn = fits.Column(name='fluxI',format='D',array=saveFluxI[:,-70:].mean(axis=1))
#dTableHdu = fits.BinTableHDU.from_columns([zeroCorrIColumn, zeroFluxIColumn])
#pHdu = fits.PrimaryHDU(header=pHeader);
#hduList = fits.HDUList([pHdu, fTableHdu, dTableHdu]);
#hduList.writeto('srh_0306_cp_zeros_' + dateName + '.fits',clobber=True);
#hduList.close();

#fluxINorm = 1/saveFluxI[:,-70:].mean(axis=1) * Zi.getSfuAtFrequency(freqList*1e-6)
#normFluxIColumn = fits.Column(name='fluxNormI',format='D',array=fluxINorm)
#dTableHdu = fits.BinTableHDU.from_columns([normFluxIColumn])
#pHdu = fits.PrimaryHDU(header=pHeader);
#hduList = fits.HDUList([pHdu, fTableHdu, dTableHdu]);
#hduList.writeto('srh_0306_cp_fluxNorm_' + dateName + '.fits',clobber=True);
#hduList.close();

fd = ftplib.FTP('10.1.1.9','sergey','jill21');

fdCPlotName = open('corrPlotName.txt','w');
fdCPlotName.write(corrPlotName_png);
fdCPlotName.close();
fi = open(corrPlotName_png,'rb');
fd.storbinary('STOR /Public/sergey/corrPlots/' + corrPlotName_png,fi);
fi.close();
fi = open('corrPlotName.txt','rb');
fd.storbinary('STOR /Public/sergey/corrPlots/corrPlotName.txt',fi);
fi.close();
fi = open('srh_0306_cp_' + dateName + '.fits','rb');
fd.storbinary('STOR /Public/sergey/corrPlots/' + 'srh_0306_cp_' + dateName + '.fits',fi);
fi.close();

fdFPlotName = open('corrMapName.txt','w');
fdFPlotName.write(fluxPlotName_png);
fdFPlotName.close();
fi = open(fluxPlotName_png,'rb');
fd.storbinary('STOR /Public/sergey/corrPlots/' + fluxPlotName_png,fi);
fi.close();
fi = open('corrMapName.txt','rb');
fd.storbinary('STOR /Public/sergey/corrPlots/corrMapName.txt',fi);
fi.close();
