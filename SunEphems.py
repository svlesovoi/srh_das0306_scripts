#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 07:15:19 2024

@author: svlesovoi
"""
def hhmmssFromNoonSec(sec):
    hh = int(sec / 3600.)
    sec -= hh*3600.
    mm = int(sec / 60.)
    sec -= mm*60
    ss = int(sec)
    return '%02d:%02d:%02d.%1d' % (hh,mm,ss,(sec - ss)*10)

def hhmmssFromDeclSec(sec):
    if (sec < 0):
        prefix = '-'
    else:
        prefix = '+'
    sec = NP.abs(sec)
    hh = int(sec / 3600.)
    sec -= hh*3600.
    mm = int(sec / 60.)
    sec -= mm*60
    ss = int(sec)
    return prefix + '%02d:%02d:%02d.%1d' % (hh,mm,ss,(sec - ss)*10)

def arcminFromDdDt(DdDt):
    if (DdDt > 0):
        return '%6.2f'%(DdDt)
    else:
        return '%6.2f'%(DdDt)
    
import numpy as NP
import datetime as DT
from BadaryRAO import BadaryRAO

days = 9
day0 = 4
now_date = DT.datetime.now()
sun_ephs = NP.zeros((days,3))
obsSun = BadaryRAO(now_date.strftime('%Y-%m-%d'),9.8,observedObject='Sun')

for day in NP.arange(0,days):
    obsSun.setDate((now_date + DT.timedelta(int(day - day0))).strftime('%Y-%m-%d'))
   
    sun_ephs[day,0] = NP.rad2deg(obsSun.declination)*3600
    sun_ephs[day,1] = obsSun.culmination
#------------------------------------------------------------------------------
for day in NP.arange(1,days-1):
    dDdT0 = (sun_ephs[day,0] - sun_ephs[day-1,0])/24
    dDdT1 = (sun_ephs[day+1,0] - sun_ephs[day,0])/24
    sun_ephs[day,2] = (dDdT0 + dDdT1)/2
    
soldat0='                  SSRT solar data'
soldat1='-----------------------------------------------------------------------'
soldat2='|year|month|day|   delta   |   noon   |radius|  p  |  b  |   l  |   d  |'
soldat3='|    |     |   |deg  m  s  | h  m  s  |  m   | deg | deg |  deg |  s/h |'
soldat4='-----------------------------------------------------------------------'
soldat5='|1992|   1 |  1|-23:03:41.0|05:14:16.3| 16.29|  2.2| -2.9| 52.02| 11.53|'

soldat_template = [soldat0, soldat1, soldat2, soldat3, soldat4, soldat5]
soldat_split = soldat5.split(soldat5[0])
 
with open('/home/svlesovoi/SSRT/soldat_' + (now_date + DT.timedelta(1)).strftime('%Y-%m-%d') + '.txt', 'w') as soldat_file:
    for soldat_line in soldat_template:
        print(soldat_line)
        soldat_file.write(soldat_line + '\r\n')
        
    for day in NP.arange(1,days-1):
        now_date_year = (now_date + DT.timedelta(int(day - day0))).strftime('%Y')
        now_date_month = (now_date + DT.timedelta(int(day - day0))).strftime('%-4m ')
        now_date_day = (now_date + DT.timedelta(int(day - day0))).strftime('%-3d')
        print(soldat5[0] + now_date_year + 
              soldat5[0] + now_date_month + 
              soldat5[0] + now_date_day + 
              soldat5[0] + hhmmssFromDeclSec(sun_ephs[day,0]) + 
              soldat5[0] + hhmmssFromNoonSec(sun_ephs[day,1]) + 
              soldat5[0] +'___.__' +
              soldat5[0] +'___._' + 
              soldat5[0] +'___._' + 
              soldat5[0] +'___.__' +
              soldat5[0] + arcminFromDdDt(sun_ephs[day,2]) +
              soldat5[0]
              )
        soldat_file.write(soldat5[0] + now_date_year + 
              soldat5[0] + now_date_month + 
              soldat5[0] + now_date_day + 
              soldat5[0] + hhmmssFromDeclSec(sun_ephs[day,0]) + 
              soldat5[0] + hhmmssFromNoonSec(sun_ephs[day,1]) + 
              soldat5[0] +'___.__' +
              soldat5[0] +'___._' + 
              soldat5[0] +'___._' + 
              soldat5[0] +'___.__' +
              soldat5[0] + arcminFromDdDt(sun_ephs[day,2]) +
              soldat5[0] + '\r\n')

