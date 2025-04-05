#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 06:32:14 2021

@author: svlesovoi
"""

import datetime as DT
from astropy.io import fits
import glob
import numpy as NP
import pylab as PL
from srhFitsFile36 import SrhFitsFile
import matplotlib
import ftplib

def hhmm_format(t, pos):
  hh = (int)(t / 3600.);
  t -= hh*3600.;
  mm = (int)(t / 60.);
  return '%02d:%02d' % (hh,mm);

dateName = DT.datetime.now().strftime("%Y%m%d")
#dateName = '20240906'
try:
    previousAntFits = fits.open('SRH0306/antPlots/srh_0306_ant_' + dateName + '.fits')
except FileNotFoundError:
    previousAntFits = 0

dt_major = 3600.;
dt_minor = 900.;

currentRawFitses = glob.glob('SRH0306/' + dateName + '/*.fit')  
currentRawFitses.sort()
currentRawFitses = NP.array(currentRawFitses[1:])

if (previousAntFits):
    previousRawFitses = previousAntFits[3].data['rawFitsNames']
    fitNames = currentRawFitses[previousRawFitses.shape[0]:]
else:
    fitNames = currentRawFitses

if len(fitNames) > 0:
    for fName in fitNames:
#        print(fName)
        sF = SrhFitsFile(fName, 512, flux_norm = False)
        if (fName == fitNames[0]):
            freqAmount  = sF.freqListLength;
            freqList = sF.freqList;
            startTime= DT.datetime.strptime(sF.hduList[0].header['DATE-OBS'] + ' ' + sF.hduList[0].header['TIME-OBS'], '%Y-%m-%d %H:%M:%S');
            times = sF.freqTime;
            antRcp = sF.ampRcp
            antLcp = sF.ampLcp
        else:
            times = NP.concatenate((times,sF.freqTime),axis=1);
            antRcp = NP.concatenate((antRcp, sF.ampRcp), axis = 1)
            antLcp = NP.concatenate((antLcp, sF.ampRcp), axis = 1)
            
        sF.close();
    
    if (previousAntFits):
        previousTime = previousAntFits[2].data['time']
        prev_freqs = previousAntFits[1].data['frequencies'].shape[0]
        prev_shape = previousAntFits[4].data['ant_RCP_00'].shape
        previousAntRcp = NP.zeros((prev_freqs,prev_shape[0],prev_shape[1]))
        previousAntLcp = NP.zeros((prev_freqs,prev_shape[0],prev_shape[1]))
        for ff in range(prev_freqs):
            previousAntRcp[ff] = previousAntFits[4].data['ant_RCP_%02d'%ff]
            previousAntLcp[ff] = previousAntFits[5].data['ant_LCP_%02d'%ff]
        saveTime = NP.concatenate((previousTime, times), axis=1)
        saveAntRcp = NP.concatenate((previousAntRcp, antRcp),axis=1)
        saveAntLcp = NP.concatenate((previousAntLcp, antLcp),axis=1)
    else:
        saveTime = times
        saveAntRcp = antRcp
        saveAntLcp = antLcp
        
    # creating antPlot
    # c_list = matplotlib.colors.LinearSegmentedColormap.from_list(PL.cm.datad['gist_rainbow'], colors=['r','g','b'], N = 16)
    # fig = PL.figure(figsize = (24,16));
    # fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05)
    # fig.tight_layout()
    # commonTitle = 'Antenna plot ' + startTime.strftime('%Y %B %d') + ', 3-6 GHz'
    # fig.suptitle(commonTitle,fontsize=14)
    # sp = fig.subplots(nrows=2,ncols=1)
    
    # for dig_rec in range(8):
    #     sp[0].cla()
    #     sp[0].set_title('RCP')
    #     sp[0].set_ylabel('arbitrary')
    #     sp[0].set_xlabel('UT')
    #     sp[0].set_ylim(-1e4,5e4)
    #     sp[0].xaxis.set_major_locator(PL.MultipleLocator(dt_major))
    #     sp[0].xaxis.set_major_formatter(PL.FuncFormatter(hhmm_format))
    #     sp[0].xaxis.set_minor_locator(PL.MultipleLocator(dt_minor))
    #     for ant in NP.arange(dig_rec*16,(dig_rec+1)*16):
    #         showAnt = saveAntRcp[:,:,ant].mean(0)
    #         sp[0].plot(saveTime[0],showAnt - showAnt.mean() + 3e3*(ant-dig_rec*16),color=c_list(ant-dig_rec*16),label=sF.antennaNames[ant])
    #     sp[0].legend(fontsize=10)
    #     sp[1].cla()
    #     sp[1].set_title('LCP')
    #     sp[1].set_ylabel('arbitrary')
    #     sp[1].set_xlabel('UT')
    #     sp[1].set_ylim(-1e4,5e4)
    #     sp[1].xaxis.set_major_locator(PL.MultipleLocator(dt_major))
    #     sp[1].xaxis.set_major_formatter(PL.FuncFormatter(hhmm_format))
    #     sp[1].xaxis.set_minor_locator(PL.MultipleLocator(dt_minor))
    #     for ant in NP.arange(dig_rec*16,(dig_rec+1)*16):
    #         showAnt = saveAntLcp[:,:,ant].mean(0)
    #         sp[1].plot(saveTime[0],showAnt - showAnt.mean() + 3e3*(ant-dig_rec*16),color=c_list(ant-dig_rec*16),label=sF.antennaNames[ant])
    #     sp[1].legend(fontsize=10)
    #     antPlotName = 'fAntPlot'+ dateName
    #     antPlotName_png = antPlotName + '_DR%02d.png'%dig_rec
    #     fig.savefig(antPlotName_png)
    
    # writing srh_ant file
        
    freqsColumn = fits.Column(name='frequencies',format='D',array=freqList)
    timeColumn = fits.Column(name='time',format=str(saveTime.shape[1]) + 'D',array=saveTime)
    antennaNamesColumn = fits.Column(name='antenna_names',format='A5',array=sF.antennaNames)
    
    rcpAntColumn = []
    lcpAntColumn = []
    for ff in range(times.shape[0]):
        rcpAntColumn.append(fits.Column(name='ant_RCP_%02d'%ff,format=str(saveAntRcp.shape[2]) + 'D',array=saveAntRcp[ff]))
        lcpAntColumn.append(fits.Column(name='ant_LCP_%02d'%ff,format=str(saveAntLcp.shape[2]) + 'D',array=saveAntLcp[ff]))
    
    fTableHdu = fits.BinTableHDU.from_columns([freqsColumn])
    tTableHdu = fits.BinTableHDU.from_columns([timeColumn])
    aTableHdu = fits.BinTableHDU.from_columns([antennaNamesColumn])
    rcpTableHdu = fits.BinTableHDU.from_columns(rcpAntColumn)
    lcpTableHdu = fits.BinTableHDU.from_columns(lcpAntColumn)
    
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
    nTableHdu = fits.TableHDU.from_columns([fitsNamesColumn])
    
    pHdu = fits.PrimaryHDU(header=pHeader)
    hduList = fits.HDUList([pHdu, fTableHdu, tTableHdu, nTableHdu, rcpTableHdu, lcpTableHdu, aTableHdu])
    hduList.writeto('SRH0306/antPlots/srh_0306_ant_' + dateName + '.fits',overwrite=True)
    hduList.close()
    
    # fd = ftplib.FTP('10.1.1.9','sergey','jill21');
    
    # fdCPlotName = open('antPlotName.txt','w');
    # fdCPlotName.write(antPlotName_png);
    # fdCPlotName.close();
    # fi = open(antPlotName_png,'rb');
    # fd.storbinary('STOR /Public/sergey/antPlots/' + antPlotName_png,fi);
    # fi.close();
    # fi = open('antPlotName.txt','rb');
    # fd.storbinary('STOR /Public/sergey/antPlots/antPlotName.txt',fi);
    # fi.close();
    # fi = open('srh_0306_ant_' + dateName + '.fits','rb');
    # fd.storbinary('STOR /Public/sergey/antPlots/' + 'srh_0306_ant_' + dateName + '.fits',fi);
    # fi.close();
    
