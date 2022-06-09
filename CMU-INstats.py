# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 12:57:05 2020

@author: William D Fahy, wdf@alumni.cmu.edu
Developed at Carnegie Mellon University
Under the supervision of Professor Ryan Sullivan 
With suggestions and input from Professor Cosma Rohilla Shalizi

Some of the code from this document was taken from a previous iteration of IN analysis.

Version 1.0

"""


'''
################################# Warning #####################################
Splineint as an interpolation method currently doesn't work with studentized
confidence intervals or comparing spectra. Since I don't use splineint anymore
I likely will not fix this, and may remove splineint entirely in a later iteration
of this code
'''

'''
There are three major sets decisions that need to be made when calculating confidencce intervals:
The first set are of course, how many simulations, what kind of simulations, and what kind of confidence intervals.
These are controlled by nSim, method, and CI variables respectively. If you're doing
a studentized CI, you also need to choose the number of resimulations. Since this 
massively increases the number of resamples (to nSim + nresim * nSim), this 
number should be kept relatively low - 50 is the default I've selected.

The second are how you're going to interpolate the resimulations and confidence intervals. By 
default, the resimulations are interpolated using the unsmoothed PCHIP algorithm and the 
confidence intervals are interpolated using an unsmoothed linear spline. Change these
at your own risk: Smoothing the CI interpolations especially can cause some weird results like 
negative k and decreasing K in the resimulations (a problem I've decided to just deal with)
or CIs crossing the actual spectrum.

These are artifacts, but still... that's no fun. I've tried to 
minimize these by artificially enforcing that CIs stay in their lane, but I may
not have caught every case.

The third decision to be made is related to the second - it's choosing the value
of num_xs. This variable determines the number of datapoints per degree used
in the smoothedPCHIP algorithm, the compareFreezes function, and in all plotting
functions. Very large numbers can make unsmoothed CIs look very complex, but 
small numbers don't lend themselves well to smoothing.

In general, if this value is high, use some sort of smoothing algorhythm for 
CIs and individual spectra. If this value is low, you probably shouldn't.

num_xs just determines the number of datapoints per degree interpolated from and plotted.
'''
num_xs = 10

import pandas as pd
import numpy as np
import sys
import os



'''
***If multiprocessing does not work or gives you an error, set MP = False!***
'''
MP = True

#You can change this to False if you don't want feedback printed while running
feedback = True

if MP:
    try:
        sys.path.index(os.path.dirname(os.path.abspath(__file__)))
    except ValueError:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        print('added to PYTHONPATH')

from scipy.stats import t
from scipy.stats import norm
from scipy.interpolate import PchipInterpolator
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import LSQUnivariateSpline

import MultiProcessingFunctions

import matplotlib.pyplot as plt
import matplotlib as mpl
import random
import copy
import multiprocessing


plt.rc('font', family='Trebuchet MS', size=30)
plt.rc('axes', linewidth=5)
plt.rc('lines', markersize=15)
plt.rc('xtick.major', size=15)
plt.rc('ytick.major', size=15)
plt.rc('ytick.minor', size=10)
plt.rc('xtick.major', width=3)
plt.rc('ytick.major', width=3)
plt.rc('ytick.minor', width=3)
plt.rc('xtick', labelsize=28)
plt.rc('ytick', labelsize=28)



'''
A freeze is an object containing a number of droplets frozen (both differential and cumulative with temperatures),
mass, mass of water, droplet volume, weight %, BET, and att.

All masses should be in g, volume is in mL, BET should be in m^2/g. 

rawSpectra is an empty variable until calcINAS is called. At that point, it calcualates
k, K, ns, nm, diffns, and diffnm without binning or interpolation (e.g. it just
uses the actual datapoints). Differential spectra are calculated such that deltaN
is 1 for each bin.

interpSpectra is an empty variable until interpSpectra is called, after which
it contains the interpolated spectra, without a fit for frozen fraction (we can't really make that continuous)

statsSpectra is created once calcCI is called. After that, it contains selected
statistics of k, K, nm, diffnm, ns, and diffns.

rawsims contains bootstrapped calculated theoretical experiments with INAS calculated
similar to rawSpectra

sims contains interpolated bootsrapped theoretical experiments

resims contain interpolated bootstrapped theoretical experiments calculated from
each of the simulations above

bkgd is a variable pointing to the freeze object that is designated as the 
pure water background of the spectrum, and bkgdSub indicates whether bkgd has been
subtracted already, preventing double subtractions.

isDiff tells the program whether or not to draw the dashed line at 1.
isbkgd should be set to True for backgrounds to indicate that there is no material 
suspended. 

The functions with repeated names within the object definition are just helper
functions that deal with some of the variable assignments within the freeze better
than doing it manually. 
'''
class Freeze(object):
    def __init__(self, freeze, data, name, att, path = None, bkgd = None):
        self.freeze = freeze
        self.name = name
        self.data = data
        self.m = float(data[0])
        self.mW = float(data[1])
        self.dV = float(data[2])
        self.suswt = float(data[3])
        self.BET = float(data[4])
        self.nD = float(data[5])
        self.bounds = None

        self.rawSpectra = None

        self.interpSpectra = None
        self.interpType = None
        self.interpStats = None

        self.CI = None

        self.sims = None
        self.rawsims = None
        self.resims = None

        self.att = att
        self.path = path

        self.bkgd = bkgd
        self.bkgdSub = False
        self.isDiff = False

        self.isbkgd = False
        
        if feedback: print("Freeze %s initialized" % self.name)

    
    '''    
    The save function currently saves a freeze as a list of temperatures
    at which the droplets present freeze along with the data necessary to interpret
    those freezing temperatures. It also maintains the attributes dictionary
    between saves. 
    
    If you have interpolated the spectrum, it then saves spectra interpreted
    as 100 datapoints and saves ns and diffns with the CIs you calculated. 
    
    If you want nm or K instead, change specType to nm or K, and it will give you
    those. 
    
    You can specify a new place to save by setting saveas to true when you call
    this function, otherwise it maintains its original path and saves it back to 
    that spot.
    
    Loading spectra will not work if you use nm or K - be aware. Maybe I'll fix this some day, maybe I won't. 
    '''
    def save(self, saveas=False, specType = 'ns'):
        if self.bkgd != None:
            self.data.append(self.bkgd.path + os.sep + self.bkgd.name + '.csv')
            if self.bkgdSub: 
                self.att.add('Background has been subtracted')
        if not self.isDiff:
            tot = pd.concat([self.freeze, pd.Series(self.freeze['nF']/self.nD, name='FF'), pd.Series(self.data, name = 'data'), pd.Series(list(self.att), name='att')],  axis='columns')
        if self.path == None:
            saveas = True
        if saveas:
            self.path= input('Enter the path to save to: ') + os.sep + self.name + '.csv'

        if self.interpStats is not None:
            keys = []
            
            if self.path.rfind(os.sep) == -1:
                cutpath = self.path[:self.path.rfind('/')]
            else:
                cutpath = self.path[:self.path.rfind(os.sep)]
            if not os.path.isdir(r'%s%sspectra' % (cutpath, os.sep)):
                os.mkdir(r'%s%sspectra ' % (cutpath, os.sep))
            
            
            xs = np.linspace(self.bounds[0], self.bounds[1], num=100)
            spectra = dict()
            spectra['Temp (degrees Celsius)'] = xs
            
            if specType == 'ns':
                keys = ['ns (Ice active sites per square centimeter)',
                        'nslowerCI (Ice active sites per square centimeter)',
                        'nsupperCI (Ice active sites per square centimeter)',
                        'diffns (Ice active sites per square centimeter degree Celsius)',
                        'diffnslowerCI (Ice active sites per square centimeter degree Celsius)',
                        'diffnsupperCI (Ice active sites per square centimeter degree Celsius)']
                for key in keys:
                    spectra[key] = np.empty(len(xs))
    
                spectra['ns (Ice active sites per square centimeter)'] = self.interpSpectra['ns'](xs)
                spectra['nslowerCI (Ice active sites per square centimeter)'] = self.interpStats['lowerCI']['ns'](xs)
                spectra['nsupperCI (Ice active sites per square centimeter)'] = self.interpStats['upperCI']['ns'](xs)
                spectra['diffns (Ice active sites per square centimeter degree Celsius)'] = self.interpSpectra['diffns'](xs)
                spectra['diffnslowerCI (Ice active sites per square centimeter degree Celsius)'] = self.interpStats['lowerCI']['diffns'](xs)
                spectra['diffnsupperCI (Ice active sites per square centimeter degree Celsius)'] = self.interpStats['upperCI']['diffns'](xs)
                spectra = pd.DataFrame.from_dict(spectra)
            elif specType == 'nm':
                keys = ['nm (Ice active sites per gram)',
                        'nmlowerCI (Ice active sites per gram)',
                        'nmupperCI (Ice active sites per gram)',
                        'diffnm (Ice active sites per gram degree Celsius)',
                        'diffnmlowerCI (Ice active sites per gram degree Celsius)',
                        'diffnmupperCI (Ice active sites per gram degree Celsius)']
                for key in keys:
                    spectra[key] = np.empty(len(xs))
    
                spectra['nm (Ice active sites per gram)'] = self.interpSpectra['nm'](xs)
                spectra['nmlowerCI (Ice active sites per gram)'] = self.interpStats['lowerCI']['nm'](xs)
                spectra['nmupperCI (Ice active sites per gram)'] = self.interpStats['upperCI']['nm'](xs)
                spectra['diffnm (Ice active sites per gram degree Celsius)'] = self.interpSpectra['diffnm'](xs)
                spectra['diffnmlowerCI (Ice active sites per gram degree Celsius)'] = self.interpStats['lowerCI']['diffnm'](xs)
                spectra['diffnmupperCI (Ice active sites per gram degree Celsius)'] = self.interpStats['upperCI']['diffnm'](xs)
                spectra = pd.DataFrame.from_dict(spectra)



            elif specType == 'K':
                keys = ['K (Ice active sites per mL)',
                        'KlowerCI (Ice active sites per mL)',
                        'KupperCI (Ice active sites per mL)',
                        'k (Ice active sites per mL degree Celsius)',
                        'klowerCI (Ice active sites per mL degree Celsius)',
                        'kupperCI (Ice active sites per mL degree Celsius)']
                for key in keys:
                    spectra[key] = np.empty(len(xs))
    
                spectra['K (Ice active sites per mL)'] = self.interpSpectra['K'](xs)
                spectra['KlowerCI (Ice active sites per mL)'] = self.interpStats['lowerCI']['K'](xs)
                spectra['KupperCI (Ice active sites per mL)'] = self.interpStats['upperCI']['K'](xs)
                spectra['k (Ice active sites per mL degree Celsius)'] = self.interpSpectra['k'](xs)
                spectra['klowerCI (Ice active sites per mL degree Celsius)'] = self.interpStats['lowerCI']['k'](xs)
                spectra['kupperCI (Ice active sites per mL degree Celsius)'] = self.interpStats['upperCI']['k'](xs)
                spectra = pd.DataFrame.from_dict(spectra)

            elif specType == 'all':
                spectraa = dict()
                spectraa['Temp (degrees Celsius)'] = xs
                spectrab = dict()
                spectrab['Temp (degrees Celsius)'] = xs
                spectrac = dict()
                spectrac['Temp (degrees Celsius)'] = xs
                
                keysa = ['ns (Ice active sites per square centimeter)',
                        'nslowerCI (Ice active sites per square centimeter)',
                        'nsupperCI (Ice active sites per square centimeter)',
                        'diffns (Ice active sites per square centimeter degree Celsius)',
                        'diffnslowerCI (Ice active sites per square centimeter degree Celsius)',
                        'diffnsupperCI (Ice active sites per square centimeter degree Celsius)']
                
                for key in keysa:
                    spectraa[key] = np.empty(len(xs))
    
                spectraa['ns (Ice active sites per square centimeter)'] = self.interpSpectra['ns'](xs)
                spectraa['nslowerCI (Ice active sites per square centimeter)'] = self.interpStats['lowerCI']['ns'](xs)
                spectraa['nsupperCI (Ice active sites per square centimeter)'] = self.interpStats['upperCI']['ns'](xs)
                spectraa['diffns (Ice active sites per square centimeter degree Celsius)'] = self.interpSpectra['diffns'](xs)
                spectraa['diffnslowerCI (Ice active sites per square centimeter degree Celsius)'] = self.interpStats['lowerCI']['diffns'](xs)
                spectraa['diffnsupperCI (Ice active sites per square centimeter degree Celsius)'] = self.interpStats['upperCI']['diffns'](xs)
                spectraa = pd.DataFrame.from_dict(spectraa)
            
                keysb = ['nm (Ice active sites per gram)',
                        'nmlowerCI (Ice active sites per gram)',
                        'nmupperCI (Ice active sites per gram)',
                        'diffnm (Ice active sites per gram degree Celsius)',
                        'diffnmlowerCI (Ice active sites per gram degree Celsius)',
                        'diffnmupperCI (Ice active sites per gram degree Celsius)']
                
                for key in keysb:
                    spectrab[key] = np.empty(len(xs))
    
                spectrab['nm (Ice active sites per gram)'] = self.interpSpectra['nm'](xs)
                spectrab['nmlowerCI (Ice active sites per gram)'] = self.interpStats['lowerCI']['nm'](xs)
                spectrab['nmupperCI (Ice active sites per gram)'] = self.interpStats['upperCI']['nm'](xs)
                spectrab['diffnm (Ice active sites per gram degree Celsius)'] = self.interpSpectra['diffnm'](xs)
                spectrab['diffnmlowerCI (Ice active sites per gram degree Celsius)'] = self.interpStats['lowerCI']['diffnm'](xs)
                spectrab['diffnmupperCI (Ice active sites per gram degree Celsius)'] = self.interpStats['upperCI']['diffnm'](xs)
                spectrab = pd.DataFrame.from_dict(spectrab)
                
                keysc = ['K (Ice active sites per mL)',
                        'KlowerCI (Ice active sites per mL)',
                        'KupperCI (Ice active sites per mL)',
                        'k (Ice active sites per mL degree Celsius)',
                        'klowerCI (Ice active sites per mL degree Celsius)',
                        'kupperCI (Ice active sites per mL degree Celsius)']
                for key in keysc:
                    spectrac[key] = np.empty(len(xs))
    
                spectrac['K (Ice active sites per mL)'] = self.interpSpectra['K'](xs)
                spectrac['KlowerCI (Ice active sites per mL)'] = self.interpStats['lowerCI']['K'](xs)
                spectrac['KupperCI (Ice active sites per mL)'] = self.interpStats['upperCI']['K'](xs)
                spectrac['k (Ice active sites per mL degree Celsius)'] = self.interpSpectra['k'](xs)
                spectrac['klowerCI (Ice active sites per mL degree Celsius)'] = self.interpStats['lowerCI']['k'](xs)
                spectrac['kupperCI (Ice active sites per mL degree Celsius)'] = self.interpStats['upperCI']['k'](xs)
                spectrac = pd.DataFrame.from_dict(spectrac)

            if specType != 'all':
                spectra.to_csv(r'%s%sspectra%s%s%s.csv' %(cutpath, os.sep, os.sep, self.name, specType), index=False)
            else:
                spectraa.to_csv(r'%s%sspectra%s%s%s.csv' %(cutpath, os.sep, os.sep, self.name, 'ns'), index=False)
                spectrab.to_csv(r'%s%sspectra%s%s%s.csv' %(cutpath, os.sep, os.sep, self.name, 'nm'), index=False)
                spectrac.to_csv(r'%s%sspectra%s%s%s.csv' %(cutpath, os.sep, os.sep, self.name, 'K'), index=False)

            if feedback: print("Saved spectra of Freeze %s to '%s\spectra\%s.csv'" %(self.name, cutpath, self.name))
        if not self.isDiff:
            tot.to_csv(r'%s' %(self.path), index=False)
            if feedback: print("Saved Freeze %s to '%s'" %(self.name, self.path))

    #Simple function to update the data variable so that saving works
    def updateData(self):
        self.data[0] = self.m
        self.data[1] = self.mW
        self.data[2] = self.dV
        self.data[3] = self.suswt
        self.data[4] = self.BET
        self.data[5] = self.nD

    #Use this function to efficiently change a single variable in an initialized freeze
    def changeVar(self, var=-99, new=None):
        if var == -99:
            var = int(input("Enter the variable you wish to change by number \n" \
                        "1: BET, 2: mass, 3: mass of water, 4: average droplet" \
                        "volume, 5: suspension wt%: "))
            new = float(input("Enter the new value for the specified variable: "))
        if var == 1: self.BET = new
        elif var == 2: self.m = new
        elif var == 3: self.mW = new
        elif var == 4: self.dV = new
        elif var == 5: self.suswt = new
        elif var == 6: self.nD = new
        else: print('No change has been made.')
        self.updateData()
        
    
    def calcINAS(self):
        if feedback: print('Calculating raw ice nucleation spectra for %s' %self.name)

        self.rawSpectra = calcINAS(self.freeze, self.dV, self.suswt, self.BET, self.nD)

        #Checks boundaries to ensure they're effective (removes first elements if they're far from the main curve, last elements if they are infinity)
        #10 is a magic number - check the first 10% to make sure there isn't a gap between them. Change if you want, this works for me.
        
        gap = 5
        #Uncomment to remove initial points if they become problematic
        
        for i in range(int(self.nD//10)):
            difftemp = self.rawSpectra['Temp'][i] - self.rawSpectra['Temp'][i+1]
            if difftemp > 1:
                gap = i+1
        
        
        if self.rawSpectra['k'].iloc[-1] == np.inf:
            endTemp = self.rawSpectra['k'].tolist().index(np.inf) - 1
        else:
            endTemp = -1


        self.bounds = (self.rawSpectra['Temp'][gap], self.rawSpectra['Temp'].iloc[endTemp])
        return self.rawSpectra
    
    def interpolateSpectra(self, interp='smoothedPCHIP'):
        if feedback: print('Interpolating ice nucleation spectra for %s' %self.name)

        self.interpSpectra = interpolateSpectra(self.rawSpectra, interp=interp, bounds=self.bounds, isbkgd=self.isbkgd)
        self.interpType = interp
        return self.interpSpectra


    def bootStats(self, nSim = 1000, method='Empirical', studentized = False):
        self.bootMethod = method
        if self.CI == 'studentized':
            studentized = True
        self.rawsims, self.sims = bootStats(self, nSim, method, interp=self.interpType, studentized = studentized)
        return self.rawsims, self.sims


    def bkgdSubtract(self):
        if feedback: print('Background subtracting %s' %self.name)
        if self.bkgdSub:
            print('Background already subtracted - manually override? (y)')
            ans = input()
            if ans != 'y':
                return
        self.interpSpectra, self.interpStats, self.sims = bkgdSubtract(self, self.bkgd, CI=self.CI)
        
    #Variations if resimulations are required for studentized CIs
    def calcCI(self, nSim = 1000, alpha = 0.05, method='Empirical', moreStats=False, CI='tskew', interp='smoothedPCHIP', nresim = 50):
        
        self.CI = CI
        if CI == 'studentized':
            self.rawsims, self.sims, self.resims = bootStats(self, nSim, method, studentized=True, interp=interp, nresim = nresim)
            self.interpStats = calcCIBody(self, self.sims, alpha=alpha, interp=self.interpType, moreStats=moreStats, CI=CI, resims=self.resims)
        else:
            self.rawsims, self.sims = bootStats(self, nSim, method, interp=interp)
            self.interpStats = calcCIBody(self, self.sims, alpha=alpha, interp=self.interpType, moreStats=moreStats, CI=CI)
        return self.interpStats

    #Simple function to plot the bootstrapped functions and the quantiles, for visualization purposes.
    def plotBoot(self, where = None, what='ns', nSim = 100):

        trash, boot = self.bootStats(nSim)
        if where == None:
            fig, where = plt.subplots(figsize=(12, 12))
        else: fig = None
        self.interpStats = calcCIBody(self, boot, interp=self.interpType, CI='quantile')

        c=mpl.colors.colorConverter.to_rgba(getColor(self), alpha=.2/(nSim/100))
        xs = np.linspace(self.bounds[0],self.bounds[1], int((self.bounds[0]-self.bounds[1])*num_xs))
        for i in range(nSim):
            if self.interpType == 'splineint' and what in ['ns', 'nm', 'K']:
                where.plot(xs, boot[i][what](self.bounds[0]) - boot[i][what](xs), ls='-', lw=1, c=c)
            else:
                where.plot(xs, boot[i][what](xs), ls='-', lw=1, c=c)

        plotSpectrum(self, where=where, what=what, mean=True, CI=True)

        #where.legend()
        where.set_ylabel(getYLabel(what))
        where.set_xlabel('Temperature (' + u"\N{DEGREE SIGN}" + 'C)')

        if what in ['K', 'ns', 'nm', 'diffns', 'diffnm', 'k']: where.set_yscale('log')
        elif what in ['FF', 'tempIntFF']: where.set_ylim(bottom=0, top=1.05)

        return fig, where


'''
For the simple calcINAS, k and K are calculated separately as is commonly
done in the field.

In this case, k is calculated using each freezing event as one bin and using
the midpoints of the distances between two events as the edges of the temperature
bins. I would argue this is better than binning by temperature.

Takes the freeze, droplet volumes, suspension weights, BET, and number of droplets.
'''
def calcINAS(freeze, dV, suswt, BET, nD):

    
    temps = freeze['Temp']
    rawSpectra = pd.DataFrame({'Temp': temps,'FF': freeze['nF']/nD, \
                         'k': [0 for i in range(len(temps))], \
                         'K': [0 for i in range(len(temps))], \
                         'nm': [0 for  i in range(len(temps))], \
                         'ns': [0 for i in range(len(temps))], \
                         'diffnm': [0 for i in range(len(temps))],\
                         'diffns': [0 for i in range(len(temps))]})

    unFrozen = pd.Series([nD for i in range(len(temps))]) - freeze['nF']

    delT=[]
    delT.append(-temps[1]+temps[0])

    for i in range(1, len(temps)-1):
        delT.append((-temps[i+1]+temps[i])/2 + (-temps[i] + temps[i-1])/2)
    delT.append(-temps.iloc[-1] - temps.iloc[-2])
    delT = pd.Series(delT)

    for i in range(len(unFrozen)):
        if unFrozen[i] <= freeze.loc[i, 'dnF']: rawSpectra.loc[i, 'k'] = np.inf
        else: rawSpectra.loc[i, 'k'] = (-np.log(1-(freeze.loc[i,'dnF']/unFrozen[i])))/(delT[i]*dV)


        if rawSpectra.loc[i, 'FF'] < 1.0:
            rawSpectra.loc[i, 'K'] = -np.log(1-rawSpectra.loc[i,'FF'])/dV
        else:
            rawSpectra.loc[i, 'K'] = np.inf
    if suswt > 0:
        rawSpectra['diffnm'] = rawSpectra['k']/suswt
        rawSpectra['diffns'] = rawSpectra['diffnm']/(BET*100**2)
        rawSpectra['nm'] = rawSpectra['K']/suswt
        rawSpectra['ns'] = rawSpectra['nm']/(BET*100**2)

    return rawSpectra

'''
Main function for calculating bootstrap simulations of the experiments. Takes the freeze,
a number of simulations, a bootstrapping method, an interpolation method, and nresim which 
is only used if interp == 'studentized'.

The empirical method is the only one implemented so far - it resamples from the 
list of freezing temperatures of droplets to estimate new spectra. 

Uses multiprocessing.

returns rawsims (basic discrete INAS spectra), sims (interpolated continuous INAS spectra),
and if the interpolation method is studentized, returns resims, the interpolated 
resimulated INAS spectra.

Individual spectra are probably best interpolated using PCHIP, but that makes the k uncertainties
quite complex. Default is smoothedPCHIP, but this can be changed if necessary. 
'''
def bootStats(freeze, nSim = 100, method='Empirical', interp='smoothedPCHIP', studentized = False, nresim=50):

    totActuallyFroze = freeze.freeze['nF'].iloc[-1]
    rawsims = []
    sims=[]
    if feedback: print('Bootstrapping %s' %freeze.name)

    if method == 'Empirical':
        simTemps = []
        tempSample = []

        for i in range(len(freeze.freeze['dnF'])):
            for j in range(int(freeze.freeze.loc[i, 'dnF'])):
                tempSample.append(freeze.freeze.loc[i, 'Temp'])
        if int(freeze.nD - totActuallyFroze) != 0:
            tempSample.extend([np.inf for i in range(int(freeze.nD - totActuallyFroze))])

        if MP == False:
            for i in range(nSim):
                simTemps = sorted(random.choices(tempSample, k=int(freeze.nD)), reverse=True)
                rawsims.append({'Temp':[], 'dnF':[]})
                for temp in simTemps:
                    if temp in rawsims[i]['Temp']:
                        rawsims[i]['dnF'][rawsims[i]['Temp'].index(temp)] += 1
                    else:
                        if temp != np.inf:
                            rawsims[i]['Temp'].append(temp)
                            rawsims[i]['dnF'].append(1)
                rawsims[i] = pd.DataFrame(rawsims[i])
                rawsims[i].insert(2, 'nF', rawsims[i]['dnF'].cumsum())

        else:
            if __name__ == '__main__':
                pool =  multiprocessing.Pool(processes=7)
                rawsims = pool.starmap(MultiProcessingFunctions.bootSim, [(tempSample, int(freeze.nD), random.random()) for i in range(nSim)])
                pool.close()
                pool.join()
        


        if studentized:
            if feedback: print('Re-bootstrapping %s for studentized confidence intervals' %freeze.name)

            reTempSample = [[] for i in range(nSim)]
            rawresims = [[] for i in range(nSim)]
            resims = [[] for i in range(nSim)]
            for i in range(nSim):

                for j in range(len(rawsims[i]['dnF'])):
                    for k in range(int(rawsims[i].loc[j, 'dnF'])):
                        reTempSample[i].append(rawsims[i].loc[j, 'Temp'])

                if MP == False:
                    for j in range(nresim):
                        simTemps = sorted(random.choices(reTempSample[i], k=int(freeze.nD)), reverse=True)
                        rawresims[i].append({'Temp':[], 'dnF':[]})
                        for temp in simTemps:
                            if temp in rawsims[i]['Temp']:
                                rawresims[i][j]['dnF'][rawresims[i][j]['Temp'].index(temp)] += 1
                            else:
                                rawresims[i][j]['Temp'].append(temp)
                                rawresims[i][j]['dnF'].append(1)
                        rawresims[i][j] = pd.DataFrame(rawresims[i][j])
                        rawresims[i][j].insert(2, 'nF', rawresims[i][j]['dnF'].cumsum())
                else:
                    if __name__ == '__main__':
                        pool = multiprocessing.Pool(processes=3)
                        rawresims[i] = pool.starmap(MultiProcessingFunctions.bootSim, [(reTempSample[i], int(freeze.nD), random.random()) for j in range(nresim)])
                        pool.close()
                        pool.join()
    else:
        print("That method has not been implemented... Yet.")
        return
    
    
    for sim in range(nSim):
        try:
            rawsims[sim] = calcINAS(rawsims[sim], freeze.dV, freeze.suswt, freeze.BET, freeze.nD)
            sims.append(interpolateSpectra(rawsims[sim], interp=interp, isbkgd=freeze.isbkgd, modS=1))
            
            #Uncomment to visualize individual sim interpolations
            # fig, where = plt.subplots(figsize=(12, 12))
            # xs = np.linspace(freeze.bounds[0],freeze.bounds[1], int((freeze.bounds[0]-freeze.bounds[1])*num_xs))
            # where.plot(xs, sims[sim]['K'](xs), ls='-', lw=1)
            # where.set_yscale('log')
        except:
            #A debugging tool - I've kept the try/except statement just in case
            print('Something went wrong with this simulation:')
            print(rawsims[sim])
        if studentized:
            for i in range(nresim):
                rawresims[sim][i] = calcINAS(rawresims[sim][i], freeze.dV, freeze.suswt, freeze.BET, freeze.nD)
                resims[sim].append(interpolateSpectra(rawresims[sim][i], interp=interp, isbkgd=freeze.isbkgd))

    if studentized:
        return rawsims, sims, resims

    return rawsims, sims

'''
This is the main function used to calculate CI for an unmodified interpolated IN
spectrum. Takes a freeze, its sims calculated from bootStats, the interpolation method,
moreStats to determine whether to return quantiles, the CI method discussed below, resims if 
CI is studentized, and plot, a variable to determine whether to plot the confidence intervals for 
testing purposes.

The posible types of CIs are as follows:
    
pivot: A method using the opposite quantiles based on theoretical background
of what CIs are. Generally poor results.

quantile: The quantiles are used as CIs. Often has good performance, easy to 
understand, but no theoretical basis.

tboot: Essentially a t-interval using the bootstrapped standard deviation. 

tskew: tbot with a skew correction

expandedquantile: Quantile confidence intervals with a skew and bias correction

studentized: The most accurate - requires resimulations to get standard deviations
for each bootstrapped spectrum, very computationally intense. For a ~300 droplet
array this takes 15-60 minutes depending on the computer. 
    
'''
def calcCIBody(freeze, sims, alpha=0.05, interp = 'smoothedPCHIP', moreStats=False, CI='tskew', resims = None, plot=False):
    
    print('Calculating confidence intervals for %s...' %freeze.name)
    
    xs = np.linspace(freeze.bounds[0],freeze.bounds[1], int((freeze.bounds[0]-freeze.bounds[1])*num_xs))

    nsim = len(sims)
    n = freeze.nD
    simPredicts = dict()
    for key in sims[0]:
        simPredicts[key] = dict()
        for i in range(len(xs)):
            simPredicts[key][i] = []
    
    for i in range(len(xs)):
        for sim in sims:
            for key in sim:
                simPredicts[key][i].append(sim[key](xs[i]))

    if CI == 'studentized':
        nresim = len(resims[0])
        resimPredicts = dict()
        for key in sims[0]:
            resimPredicts[key] = dict()
            for sim in range(nsim):
                resimPredicts[key][sim] = dict()
                for i in range(len(xs)):
                    resimPredicts[key][sim][i] = []
                    for j in range(nresim):
                        resimPredicts[key][sim][i].append(resims[sim][j][key](xs[i]))

        resimPredicts = pd.DataFrame.from_dict({(i, j, k): resimPredicts[i][j][k] for i in resimPredicts.keys() for j in resimPredicts[i].keys() for k in resimPredicts[i][j].keys()}, dtype = float)


    simMean = {i:[] for i in simPredicts.keys()}
    simStdev = {i:[] for i in simPredicts.keys()}
    simUpperCI = {i:[] for i in simPredicts.keys()}
    simLowerCI = {i:[] for i in simPredicts.keys()}
    samplemeanlst = {i:[] for i in simPredicts.keys()}
    if moreStats:
        simUpperQuantile = {i:[] for i in simPredicts.keys()}
        simLowerQuantile = {i:[] for i in simPredicts.keys()}
    simPredicts = pd.DataFrame.from_dict({(i,j): simPredicts[i][j] for i in simPredicts.keys() for j in simPredicts[i].keys()}, dtype = float)

    for i in range(len(xs)):
        for key in sims[0]:
            samplemean = freeze.interpSpectra[key](xs[i])
            samplemeanlst[key].append(samplemean)
            mean = simPredicts[key, i].mean(axis=0, skipna=True)
            stdev = simPredicts[key, i].std(axis=0, skipna=True)
            simMean[key].append(mean)
            simStdev[key].append(stdev)
            if moreStats:
                simLowerQuantile[key].append(simPredicts[key, i].quantile(q=alpha/2))
                simUpperQuantile[key].append(simPredicts[key, i].quantile(q=1-alpha/2))


            #Standard deviations and skew measurements are *ALREADY NORMALIZED* by N-1, and as such are used without modification as per Hesterberg (2014)
            if CI == 'pivot':
                simLowerCI[key].append(2*samplemean - simPredicts[key, i].quantile(q=1-alpha/2))
                simUpperCI[key].append(2*samplemean - simPredicts[key, i].quantile(q=alpha/2))
            elif CI == 'quantile':
                simLowerCI[key].append(simPredicts[key, i].quantile(q=alpha/2) + samplemean - mean)
                simUpperCI[key].append(simPredicts[key, i].quantile(q=1-alpha/2) + samplemean - mean)
            elif CI == 'tboot':
                simLowerCI[key].append(samplemean + t.ppf(alpha/2, n-1)*stdev)
                simUpperCI[key].append(samplemean + t.ppf(1-alpha/2, n-1)*stdev)
            elif CI == 'tskew':
                kappa = simPredicts[key,i].skew(axis=0, skipna=True)/(6*np.sqrt(n))
                simLowerCI[key].append(samplemean+(t.ppf(alpha/2, n-1) + kappa*(1 + 2*(t.ppf(alpha/2, n-1)**2)))*stdev)
                simUpperCI[key].append(samplemean+(t.ppf(1-alpha/2, n-1) + kappa*(1 + 2*(t.ppf(1-alpha/2, n-1)**2)))*stdev)
            elif CI == 'expandedquantile':
                modalpha = norm.cdf(-np.sqrt(n/(n-1)) * -t.ppf(alpha/2, n-1))
                simLowerCI[key].append(simPredicts[key, i].quantile(q=modalpha/2) + samplemean - mean)
                simUpperCI[key].append(simPredicts[key, i].quantile(q=1-modalpha/2) + samplemean - mean)
            elif CI == 'studentized':
                resimstdev = []
                q = []
                for j in range(nsim):
                    resimstdev.append(resimPredicts[key, j, i].std(axis=0, skipna=True))
                    if resimstdev[j] == 0:
                        if simPredicts[key, i][j] == mean:
                            q.append(0)
                        elif simPredicts[key, i][j] - mean < 0:
                            q.append(-np.inf)
                        else:
                            q.append(np.inf)
                    else:
                        q.append((simPredicts[key, i][j]-mean)/resimstdev[j])
                simLowerCI[key].append(samplemean - stdev*np.quantile(q, 1-alpha/2))
                simUpperCI[key].append(samplemean - stdev*np.quantile(q, alpha/2))


    simMean = pd.DataFrame(simMean)
    simStdev = pd.DataFrame(simStdev)
    simLowerCI = pd.DataFrame(simLowerCI)
    simUpperCI = pd.DataFrame(simUpperCI)
    samplemeanlst = pd.DataFrame(samplemeanlst)

    if freeze.isbkgd:
        intkeys = ['K']
    else:
        intkeys = ['ns', 'nm', 'K']
    
    '''
    Special case - splineint results in a negative slope
    
    Corrects confidence intervals to ensure there are no invalid results
    '''
    
    if interp == 'splineint':
        for i in range(1, len(xs)):
            for key in intkeys:
                if simLowerCI.loc[i,key] > simLowerCI.loc[i-1,key]:
                    simLowerCI.loc[i,key]= simLowerCI.loc[i-1,key]
                if simUpperCI.loc[i,key] > simUpperCI.loc[i-1,key]:
                    simUpperCI.loc[i,key] = simUpperCI.loc[i-1,key]
            simLowerCI[simLowerCI<samplemeanlst] = samplemeanlst[simLowerCI<samplemeanlst]
            simLowerCI[simLowerCI<0] = 0
            simUpperCI[simUpperCI<0] = 0
            simUpperCI[simUpperCI>samplemeanlst] = samplemeanlst[simUpperCI>samplemeanlst]
    else:
        simLowerCI[simLowerCI<0] = 0
        simLowerCI[simLowerCI>samplemeanlst] = samplemeanlst[simLowerCI>samplemeanlst]
        simUpperCI[simUpperCI<0] = 0
        simUpperCI[simUpperCI<samplemeanlst] = samplemeanlst[simLowerCI<samplemeanlst]
        for i in range(1, len(xs)):
            for key in intkeys:
                if simLowerCI.loc[i,key] < simLowerCI.loc[i-1,key]:
                    simLowerCI.loc[i,key] = simLowerCI.loc[i-1,key]
                if simUpperCI.loc[i,key] < simUpperCI.loc[i-1,key]:
                    simUpperCI.loc[i,key] = simUpperCI.loc[i-1,key]

    if plot:
        fig, where = makePlot([freeze], what='k')
        fig1, where1 = makePlot([freeze], what='K')
        where.plot(xs, simLowerCI['k'])
        where.plot(xs, simUpperCI['k'])
        where1.plot(xs, simLowerCI['K'])
        where1.plot(xs, simUpperCI['K'])


    simMean.insert(0, 'Temp', xs)
    simStdev.insert(0, 'Temp', xs)
    simUpperCI.insert(0, 'Temp', xs)
    simLowerCI.insert(0, 'Temp', xs)
    if moreStats:
        simLowerQuantile = pd.DataFrame(simLowerQuantile)
        simUpperQuantile = pd.DataFrame(simUpperQuantile)
        simLowerQuantile.insert(0, 'Temp', xs)
        simUpperQuantile.insert(0, 'Temp', xs)

    interpStats = dict()

    interpStats['mean'] = interpolateWellDefined(simMean, interp=interp)
    interpStats['stdev'] = interpolateWellDefined(simStdev, interp=interp)
    interpStats['upperCI'] = interpolateWellDefined(simUpperCI, interp=interp)
    interpStats['lowerCI'] = interpolateWellDefined(simLowerCI, interp=interp)
    if moreStats:
        interpStats['upperQuantile'] = interpolateWellDefined(simUpperQuantile, interp=interp)
        interpStats['lowerQuantile'] = interpolateWellDefined(simLowerQuantile, interp=interp)
    
    
    #Uncomment to visualize the confidence interval data points vs. the interpolation
    '''
    fig, where = plt.subplots(figsize=(12, 12))
    xs = np.linspace(freeze.bounds[0],freeze.bounds[1], int((freeze.bounds[0]-freeze.bounds[1])*num_xs))
    where.plot(xs, interpStats['upperCI']['k'](xs), ls='-', lw=3)
    where.plot(xs, interpStats['lowerCI']['k'](xs), ls='-', lw=3)
    where.plot(simUpperCI['Temp'], simUpperCI['k'], marker='o', lw=0)
    where.plot(simLowerCI['Temp'], simLowerCI['k'], marker='o', lw=0)
    where.set_ylim([1e1, 1e5])
    where.set_yscale('log')
    '''
    
    return interpStats

def interpolateSpectra(rawspectrum, interp='smoothedPCHIP', bounds=None, isbkgd = False, modS=1):    
    
    if isbkgd:
        diffkeys = ['k']
        intkeys = ['K']
    else:
        diffkeys = ['k', 'diffnm', 'diffns']
        intkeys = ['K', 'nm', 'ns']

    if bounds != None:
        rawspectrum = rawspectrum.iloc[rawspectrum[rawspectrum['Temp'] == bounds[0]].index[0]:]
    
    rawspectrumsorted = rawspectrum
    rawspectrumsorted = rawspectrumsorted.replace([np.inf, -np.inf], np.nan)
    rawspectrumsorted = rawspectrumsorted.dropna()
    rawspectrumsorted = rawspectrumsorted.sort_values(by='Temp')

    if bounds == None:
        bounds = [rawspectrumsorted['Temp'].iloc[-1], rawspectrumsorted['Temp'].iloc[0]]
    ub = bounds[0]
    
    if interp == 'bins':
        print('Just use the other file please....')

    if interp == 'smoothedPCHIP':
        xs = np.linspace(bounds[1], bounds[0], int((bounds[0]-bounds[1])*num_xs))
        newspec = dict()
        oldspl = dict()
        spl = dict()
        invspl = dict()
        t = None
        for key in intkeys:
            oldspl[key] = PchipInterpolator(rawspectrumsorted['Temp'], rawspectrumsorted[key], extrapolate=False)
            
            newspec[key] = np.asarray([oldspl[key](i) for i in xs])
            newspec['Temp'] = xs
            
            #Ensures scale invariance between normalized spectra by making the knots identical
            if t is None:
                spl[key] = UnivariateSpline(newspec['Temp'], newspec[key], s = abs(modS*len(newspec[key])*np.nanmean(newspec[key]))/2000, w=1/(1+np.sqrt(abs(newspec[key]))), ext=3)
                t = spl[key].get_knots()
            else:
                spl[key] = LSQUnivariateSpline(newspec['Temp'], newspec[key], t[1:-1], w=1/(1+np.sqrt(abs(newspec[key]))), ext=3)
            invspl[key] = LSQUnivariateSpline(newspec['Temp'], -newspec[key], t[1:-1], w=1/(1+np.sqrt(abs(newspec[key]))), ext=3)

            
        spl['k'] = invspl['K'].derivative()
        
        #Uncomment to visualize an interpolated curve vs its data points
        '''      
        fig, where = plt.subplots(figsize=(12, 12))
        where.plot(xs, spl['K'](xs), ls='-', lw=3)
        where.plot(newspec['Temp'], newspec['K'], marker='o', lw=0)
        where.set_ylim([1e1, 1e5])
        where.set_yscale('log')
        '''
        
        if not isbkgd:
            spl['diffnm'] = invspl['nm'].derivative()
            spl['diffns'] = invspl['ns'].derivative()
        return spl


    if interp == 'PCHIP':
        spl = dict()
        invspl = dict()
        for key in intkeys:

            spl[key] = PchipInterpolator(rawspectrumsorted['Temp'], rawspectrumsorted[key], extrapolate=False)
            invspl[key] = PchipInterpolator(rawspectrumsorted['Temp'], -rawspectrumsorted[key], extrapolate=False)

        spl['k'] = invspl['K'].derivative()
        if not isbkgd:
            spl['diffnm'] = invspl['nm'].derivative()
            spl['diffns'] = invspl['ns'].derivative()

        return spl

    #Main function, allows conversion from K to k without a problem
    if interp == 'splinederiv':
        spl = dict()
        invspl = dict()
        for key in intkeys:

            spl[key] = UnivariateSpline(rawspectrumsorted['Temp'], rawspectrumsorted[key], w=1/np.sqrt(rawspectrumsorted[key]), ext=3)
            invspl[key] = UnivariateSpline(rawspectrumsorted['Temp'], -rawspectrumsorted[key], w=1/np.sqrt(rawspectrumsorted[key]), ext=3)
        spl['k'] = invspl['K'].derivative()
        if not isbkgd:
            spl['diffnm'] = invspl['nm'].derivative()
            spl['diffns'] = invspl['ns'].derivative()

        return spl

    if interp == 'splineint':
        spl = dict()
        splinv = dict()

        for key in diffkeys:
            spl[key] = UnivariateSpline(rawspectrumsorted['Temp'], rawspectrumsorted[key], s=200*rawspectrumsorted[key].iloc[-1]*len(rawspectrumsorted[key]), w=1/np.sqrt(rawspectrumsorted[key]))
            splinv[key] = UnivariateSpline(rawspectrumsorted['Temp'], rawspectrumsorted[key], s=200*rawspectrumsorted[key].iloc[-1]*len(rawspectrumsorted[key]), w=1/np.sqrt(rawspectrumsorted[key]))

        spl['K'] = splinv['k'].antiderivative()
        if not isbkgd:
            spl['nm'] = splinv['diffnm'].antiderivative()
            spl['ns'] = splinv['diffns'].antiderivative()

        return spl

    #Alright tbh
    if interp == 'chebyshevint':
        spl = dict()
        invspl = dict()

        for key in diffkeys:
            spl[key] = np.polynomial.Chebyshev.fit(rawspectrumsorted['Temp'], rawspectrumsorted[key], 4, w=1/abs(rawspectrumsorted[key]**(3/5)))
            invspl[key] = np.polynomial.Chebyshev.fit(rawspectrumsorted['Temp'], -rawspectrumsorted[key], 4, w=1/abs(rawspectrumsorted[key]**(3/5)))


        spl['K'] = invspl['k'].integ(lbnd=ub)
        if not isbkgd:
            spl['ns'] = invspl['diffns'].integ(lbnd=ub)
            spl['nm'] = invspl['diffnm'].integ(lbnd=ub)
        return spl

    if interp == 'powerint':
        spl = dict()
        invspl = dict()

        for key in diffkeys:
            spl[key] = np.polynomial.Polynomial.fit(rawspectrumsorted['Temp'], rawspectrumsorted[key], [0,1,2,3,4], w=1/abs(rawspectrumsorted[key]**(1/2)))
            invspl[key] = np.polynomial.Polynomial.fit(rawspectrumsorted['Temp'], -rawspectrumsorted[key], [0,1,2,3,4], w=1/abs(rawspectrumsorted[key]**(1/2)))


        spl['K'] = invspl['k'].integ(lbnd=ub)
        if not isbkgd:
            spl['ns'] = invspl['diffns'].integ(lbnd=ub)
            spl['nm'] = invspl['diffnm'].integ(lbnd=ub)
        return spl

'''
To be called when separate interpolations are needed for k and K (e.g. when 
interpolating confidence intervals). 

Simple linear spline fit for when I have an extremely dense grid of temperature data
Note: Currently results in slightly different CIs for spectra that should just be
scaled versions of each other. This is a bug, but I'm not sure if it's worth fixing,
as we'd have to give up the true interpolation of this function.
'''
def interpolateWellDefined(spectrum, interp=None, modS = 1):
    rawspectrumsorted = spectrum
    rawspectrumsorted = rawspectrumsorted.replace([np.inf, -np.inf], np.nan)
    rawspectrumsorted = rawspectrumsorted.dropna()
    rawspectrumsorted = rawspectrumsorted.sort_values(by='Temp')
    spl = dict()
    for key in spectrum.keys():
        spl[key] = InterpolatedUnivariateSpline(rawspectrumsorted['Temp'],rawspectrumsorted[key], k=1, ext=3)
    return spl



'''
InitFreeze should be called to initialize freezes into the python format, but
after that point, data is stored and so experiments or freezes should be loaded
directly.

Names are entered for volcanic ash using the following code:
FUE_YYMMDD_OG_ON5_1_2
(ash code, date, chemical treatment, physical treatement, iteration #, freeze #)

Each of these strings separated by '_' will be searchable to plot, 
but you can add extras using the attribute input.

'''
def initFreeze(path = None, freezes = None):
    #Change this 'cal' variable if you would like to calibrate data, otherwise leave it alone
    '''
    0.92257, -1.2839  3/12/18
	0.9265. -1.0553 //3/14/18
	0.9641, -0.0732 08/21/18
	0.9207,  0.1755 10/30/18
    0.946, -0.8406 09/20/19
    0.9269, -1.1899 After (most are already calibrated though)
    '''
    cal = False
    slope = 1
    intercept = -0.8406

    if path == None: path = input("enter the path to the input excel file: ")

    try: freeze = pd.read_excel(path)
    except:
        print('Invalid path')
        return

    nD = len(freeze['Radii'])




        


    freeze.rename(columns={'Frozen Fraction':'nF', 'Temperature':'Temp'}, inplace=True)

    if cal:
        freeze['Temp'] = freeze['Temp']*slope + intercept

    freeze.drop(columns=['Freezing Darkness', 'Original Darkness', 'Radius', 'Radii'], inplace=True)

    freeze['nF'] = freeze['nF']*nD

    freeze.insert(1, 'dnF', freeze['nF'])

    for i in range(len(freeze['dnF'])-1, 1, -1):
        freeze.loc[i, 'dnF'] = freeze.loc[i, 'dnF'] - freeze.loc[i-1, 'dnF']

    freeze.dropna(inplace=True)

    print(freeze.head())
    
    #divided by 1000 to get mL
    dV = float(input('Enter the volume of each droplet in microliters'))/1000
    m = float(input('Enter the mass of sample in the freeze in g: '))
    mW = float(input('Enter the mass of water in the freeze in g: '))
    BET = float(input('Enter the BET of the sample in m^2/g: '))
    name = input('Enter the name of this freeze: ')
    suswt = m/mW
    att = set(input('Enter any attributes of the sample to search by, separated by commas: ').split(','))

    att.update(name.split('_'))

    out = Freeze(freeze, [m, mW, dV, suswt, BET, nD], name, att)

    out.calcINAS()

    if freezes != None: freezes[name] = out

    return out


'''If you use this, you need to recalculate EVERYTHING. It adjusts a calibration
if you made a serious mistake.'''
def recal(freeze, oldcal, newcal):
    freeze.freeze['Temp'] = ((freeze.freeze['Temp']-oldcal[1])/oldcal[0])*newcal[0] + newcal[1]
    freeze.calcINAS()




'''******************************************************************PLOTTING***********************************************'''

'''
These are the style functions used in the plotting functions. Change them as needed.

Colors are taken from here:
https://davidmathlogic.com/colorblind/#%23332288-%23117733-%2344AA99-%2388CCEE-%23DDCC77-%23CC6677-%23AA4499-%23882255
'''

def getLine(freeze):
    return '-'



def getMarker(freeze):
    return 'o'


def getColor(freeze):
    
    if freeze.CI == 'pivot':
        return '#CC6677'
    elif freeze.CI == 'quantile':
        return '#DDCC77'
    elif freeze.CI == 'tboot':
        return '#88CCEE'
    elif freeze.CI == 'tskew':
        return '#AA4499'
    elif freeze.CI == 'expandedquantile':
        return '#332288'
    elif freeze.CI == 'studentized':
        return '#117733'
    
    
    if freeze.name == 'A':
        return '#882255'
    elif freeze.name == 'B':
        return '#CC6677'
    elif freeze.name == 'C':
        return '#DDCC77'
    return '#88CCEE'
    
    
    if 'Combined' in freeze.att:
        if 'OG' in freeze.att:
            return '#117733'
        elif 'b1' in freeze.att:
            return 'black'
        return '#88CCEE'
    
    if 'PW' in freeze.name: 
        if freeze.name[18] == '1':
            return adjust_lightness('#882255', float(freeze.name[20])/4 + 1.5)
        else:
            return adjust_lightness('#88CCEE', float(freeze.name[20])/4 + 0.41)
    return adjust_lightness('#117733', float(freeze.name[20])/4 + 1)
    
    if freeze.interpType == 'PCHIP':
        return '#332288'
    elif freeze.interpType == 'splineint':
        return '#CC6677'
    elif freeze.interpType == 'splinederiv':
        return '#882255'
    elif freeze.interpType == 'smoothedPCHIP':
        return '#117733'
    return '#88CCEE'

    if freeze.name[-13] == '1':
        return adjust_lightness('#882255', float(freeze.name[-11])/4 + 1.5)
    else:
        return adjust_lightness('#88CCEE', float(freeze.name[-11])/4 + 0.41)

    return getRandomColor()

def getLabel(freeze, what):
    if freeze.name == 'A':
        return 'nSim = 100'
    elif freeze.name == 'B':
        return 'nSim = 1000'
    elif freeze.name == 'C':
        return 'nSim = 10000'
    
    return 'N = %d' % freeze.nD

    if 'Combined' in freeze.att:
        if 'OG' in freeze.att:
            return 'Unaged FUE'
        elif 'b1' in freeze.att:
            return 'Filtered water background'
        return 'Water aged FUE'
    
    if 'PW' in freeze.name:
        return 'Water aged FUE %s|%s' %(freeze.name[18], freeze.name[20])
    return 'Unaged FUE %s' %freeze.name[20]
    
    return 'Water aged FUE compared to unaged FUE'
    
    return freeze.CI

    if freeze.name=='PWAD5':
        return 'Water aged FUE'
    return 'Unaged FUE'

    return freeze.name


def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def plotSpectrum(freeze, what='ns', raw=False, where=None, c=None, CI=False, mean=False, alsoraw=False):
    if where == None:
        fig, where = plt.subplots(figsize=(12, 12))
    else: fig = None

    if c == None:
        c=getColor(freeze)

    if raw or alsoraw or what == 'FF':
        where.plot(freeze.rawSpectra['Temp'], freeze.rawSpectra[what], ls='-', lw=0,c=c, mec=c, mfc='none', \
                   mew=3, marker = getMarker(freeze), label=getLabel(freeze, what), zorder=2, alpha=1)

    if alsoraw or not raw and what != 'FF':
        xs = np.linspace(freeze.bounds[0],freeze.bounds[1], int((freeze.bounds[0]-freeze.bounds[1])*num_xs))
        if CI:
            if freeze.interpType == 'splineint' and what in ['ns', 'nm', 'K']:
                where.plot(xs, freeze.interpStats['upperCI'][what](freeze.bounds[0]) - freeze.interpStats['upperCI'][what](xs), ls=':', lw=1.5, c=c)
                where.plot(xs, freeze.interpStats['lowerCI'][what](freeze.bounds[0]) - freeze.interpStats['lowerCI'][what](xs), ls=':', lw=1.5, c=c)
            else:
                where.plot(xs, freeze.interpStats['upperCI'][what](xs), ls='dashed', lw=3, c=c)
                where.plot(xs, freeze.interpStats['lowerCI'][what](xs), ls='dashed', lw=3, c=c)

        if freeze.interpType == 'splineint' and what in ['ns', 'nm', 'K']:
            where.plot(xs, freeze.interpSpectra[what](freeze.bounds[0]) - freeze.interpSpectra[what](xs), ls=getLine(freeze), lw=5, c=c, label=getLabel(freeze, what))
        else:
            if mean:
                where.plot(xs, freeze.interpStats['mean'][what](xs), ls=getLine(freeze), lw=5, c=c, label=getLabel(freeze, what))
            else:
                where.plot(xs, freeze.interpSpectra[what](xs), ls=getLine(freeze), lw=5, c=c, label=getLabel(freeze, what))
    return fig



def makePlot(freezes, what=None, where=None, raw=False, CI=False, mean=False, scale='auto', alsoraw=False, legend=True):

    newfig = False

    if where == None:
        fig, where = plt.subplots(figsize=(12, 12))
        newfig = True


    if what is None: what = input('Enter what you would like to plot')


    if not isinstance(freezes, list):
        freezes = [freezes]
    
    hit = -50
    lot = 0

    for freeze in freezes:

        hit = max(hit, freeze.bounds[0])
        lot = min(lot, freeze.bounds[1])


    if freezes[0].isDiff:
        if freezes[0].diffMethod == 'divide':
            where.plot([lot-1, hit+1], [1,1], lw=5, ls='dashed', c='k')
        else:
            where.plot([lot-1, hit+1], [0,0], lw=5, ls='dashed', c='k')
            
    for freeze in freezes: plotSpectrum(freeze, where=where, what=what, raw=raw, CI=CI, mean=mean, alsoraw=alsoraw)
    
    if legend == 'external':
        if len(freezes) >=9:
            where.legend(ncol=4, framealpha=1)
        else:
            where.legend(ncol = 3, framealpha=1)
    elif legend:
        where.legend(framealpha=0)
        
    if freezes[0].isDiff:
        if freezes[0].diffMethod == 'divide':
            where.set_ylabel('Ratio of ' + getYLabel(what))
        elif freezes[0].diffMethod == 'subtract':
            where.set_ylabel('Difference in ' + getYLabel(what))
    else:
        where.set_ylabel(getYLabel(what))
    where.set_xlabel('Temperature (' + u"\N{DEGREE SIGN}" + 'C)')


    if what in ['K', 'ns', 'nm', 'diffns', 'k', 'diffnm']:
        if not freezes[0].isDiff or freezes[0].diffMethod == 'divide':
            pass
            where.set_yscale('log')
    elif what in ['FF', 'tempIntFF']: where.set_ylim(bottom=0, top=1)

    if scale == 'auto':
        where.set_xlim(left=lot-1, right=hit+1)
        if not freezes[0].isDiff:
            where.set_ylim(freeze.rawSpectra.iloc[freeze.rawSpectra[freeze.rawSpectra['Temp'] == freeze.bounds[0]].index[0]][what]*.1, freeze.rawSpectra.iloc[freeze.rawSpectra[freeze.rawSpectra['Temp'] == freeze.bounds[1]].index[0]][what]*10)
    else:
        where.set_xlim(left=scale[2], right=scale[3])
        where.set_ylim(bottom=scale[0], top=scale[1])
    
    if newfig:
        return fig, where
    

def getYLabel(what):
    if what == 'K':
        return 'Ice active sites per mL'
    elif what == 'k':
        return 'Ice active sites per mL ' +  u"\N{DEGREE SIGN}" + 'C'
    elif what == 'nm': return 'Ice active sites per g'
    elif what =='ns': return 'Ice active sites per $cm^2$'
    elif what == 'diffnm': return 'Ice active sites per g ' +  u"\N{DEGREE SIGN}" + 'C'
    elif what == 'diffns': return 'Ice active sites per $cm^2$ '  +  u"\N{DEGREE SIGN}" + 'C'
    elif what == 'FF' or what == 'tempIntFF': return 'Frozen fraction'
    else: return 'Nucleation Coordinate'

def getRandomColor():
    return random.choice(['#332288', '#117733', '#44AA99', '#88CCEE', '#DDCC77', '#CC6677', '#AA4499', '#882255'])

'''*********************************************************COMBINING****************************************************'''

'''
An experiment is an object containing an arbitrary number of freezes combined
into a single frozen fraction and a list of pointers to the spectra included.
It should be initialized either by loading a known experiment file or by
using the 'combineFreezes' function, never by hand
'''
class Exp(Freeze):
    def __init__(self, FF, data, name, att, freezes, path=None, bkgd=None):
        super().__init__(FF, data, name, att, path, bkgd)
        self.freezes = freezes
        self.num = len(freezes)

class diffExp(Exp):
    def __init__(self, spectra, stats, sims, data, name, att, freezes, diff, path=None, bkgd=None):
        super().__init__(None, data, name, att, freezes, path, bkgd)
        self.interpSpectra = spectra
        self.interpStats = stats
        self.sims = sims
        self.isDiff = True
        self.diffMethod = diff

#Utility function to select freezes from freezes in with attributes atts
def selectFreezes(atts, freezesin, freezesout=None):
    if freezesout == None: freezesout=[]

    for freeze in freezesin:
        incurr = True
        for attr in atts:

            if attr not in freezesin[freeze].att:
                incurr = False
        if incurr: freezesout.append(freezesin[freeze])
    return freezesout

'''
Function to combine a group of freezes into a single 'experiment'. Note that
this requires the assumption that all the droplets are drawn from a single
population, e.g. that the solutions are all the same. In practice this is never
true because of slight variations in samples, wt%s, etc. Be mindful when
you're using this.

This doesn't require INAS to be calculated.
'''
def combineLikeFreezes(freezes, name=None):
    
    
    if feedback: print("combining the list of freezes starting with %s" %freezes[0].name)
    
    nF, Temp = [], []
    num = len(freezes)

    c = [0 for i in range(num)]

    while not all([c[i] == -1 for i in range(num)]):
        minT = -50
        frozen = 0
        freezeNum = 0

        for i in range(num):
            if c[i] != -1 and (freezes[i].freeze['Temp'][c[i]] >= minT):
                minT = freezes[i].freeze['Temp'][c[i]]
                frozen = freezes[i].freeze['nF'][c[i]]
                freezeNum = i

        if minT in Temp:
            idx = Temp.index(minT)
            if c[freezeNum] == 0: nF[idx] += frozen
            elif c[freezeNum] == -1: print('How?')
            else: nF[idx] += frozen - freezes[freezeNum].freeze['nF'][c[freezeNum]-1]
        else:
            if c[freezeNum] == 0: nF.append(frozen)
            elif c[freezeNum] == -1: print('How?')
            else: nF.append(frozen - freezes[freezeNum].freeze['nF'][c[freezeNum]-1])
            Temp.append(minT)
        c[freezeNum] += 1
        if c[freezeNum] >= len(freezes[freezeNum].freeze): c[freezeNum] = -1

    dnF = pd.Series(nF)
    Temp = pd.Series(Temp)

    nF = dnF.cumsum()


    totAtt = set()
    m, mW, BET, dV, suswt = 0, 0, 0, 0, 0

    for freeze in freezes:
        m += freeze.m
        mW += freeze.mW
        suswt += freeze.suswt
        BET += freeze.BET
        dV += freeze.dV
        totAtt.update(freeze.att)

    m /= num
    mW /= num
    dV /= num
    BET /= num
    suswt /= num
    nD = nF.iloc[-1]

    joined = pd.DataFrame({'Temp':Temp, 'dnF':dnF, 'nF':nF})

    if name == None:
        name = input('Enter the name of this combined freeze: ')

    out = Exp(joined, [m, mW, dV, suswt, BET, nD], name, totAtt, freezes)

    out.calcINAS()
    out.interpolateSpectra()

    return out

'''
Function to subtract a background (bkgd) from a freeze, bootstrapping CIs 
and re-calculating bootstrapped spectra so that the background subtracted samples
can be used in compareFreezes.
'''
def bkgdSubtract(freeze, bkgd, method='Empirical', CI='tskew'):
    if feedback: print('Calculating background subtracted spectrum for %s' %freeze.name)
    actualDiff, interpStats, sims, trash= bootstrapFreezeDiff(freeze, bkgd, CI=CI, diff = 'subtract', interp=freeze.interpType, bkgd = True)

    return actualDiff, interpStats, sims

'''
Wrapper function for bootstrapFreezeDiff, where ref is the untreated or control sample
and samp is the experimental sample. If the INA of samp is less than ref, then the difference will 
be less than 1 if dividing and less than 0 if subtracting, to represent that samp is lower than ref.
'''
def compareFreezes(samp, ref, nSim=None, method='Empirical', CI='tskew', diff = 'divide', interp=None):
    
    print('comparing %s with %s' %(samp.name, ref.name))
    
    if interp == None:
        interp = ref.interpType
    actualDiff, newStats, sims, bounds = bootstrapFreezeDiff(samp, ref, nSim=nSim, CI=CI, diff=diff, interp = interp)

    out = diffExp(actualDiff, newStats, sims, copy.deepcopy(samp.data), samp.name + 'Difference', copy.deepcopy(samp.att), [ref, samp], diff, bkgd = samp.bkgd)
    out.bounds = bounds
    return out



'''
Main function to subtract/divide two freezes with bootstrapping for CIs. 

Returns actualDiff, the interpolated differences, interpstats, interpolated
statistics on the difference, sims, the differences between pairwise simulations 
for, and bounds, describing the boundaries of the difference samples (usually the overlap) 
                                                                      
Takes samp (the experimental sample), ref (the freeze to be subtracted/divided, or a background),
nSim, the number of sims, CI, the type of CIs used, diff, either divide or subtract,
interp, the interpolation method, alpha, the confidence leverl, bkgd, to be used
if ref is a background so that only k and K are directly compared, and moreStats, a boolean
that causes the function to return quantiles as well as CIs if True.

If simulations or the experiment have the refence value as being 0 (in differential spectra),
then the divided result is nan, and will be ignored for the purposes of interpolation
and statistics. 

Has some hard-coded limits to ensure the CIs don't cross the actual value, but 
I can't promise all cases are caught. 
'''
def bootstrapFreezeDiff(samp, ref, nSim=None, CI='tskew', diff = 'divide', interp='smoothedPCHIP', alpha = 0.05, bkgd = False, moreStats=False):
    n = ref.nD + samp.nD
    if samp.CI != ref.CI:
        if samp.CI == 'studentized' or ref.CI == 'studentized':
            raise(ValueError('Fatal mismatched CI error'))
        else:
            print('Warning, nonfatal mismatched CI methods \n')
            ans = input('Enter 1 if you would like to continue')
            if ans != '1':
                raise(ValueError('Mismatched CI error'))
    if not type(ref.sims) is list or not type(ref.sims) is list or len(ref.sims) != len(samp.sims):
        print('Please calculate CIs before comparing spectra')
        return [None, None, None, None]
            
    nSim = len(samp.sims)
    sims = [None for i in range(nSim)]

    if bkgd:
        upper = samp.bounds[0]
        lower = samp.bounds[1]
    else:
        upper = min(ref.bounds[0], samp.bounds[0])
        lower = max(ref.bounds[1], samp.bounds[1])

    try:
        xs = np.linspace(upper, lower, int((upper-lower)*num_xs))
    except:
        print('These spectra do not overlap, and therefore cannot be compared.')
        return [None, None, None, None]
        
    simPredicts = dict()
    actualDiff = dict()
    for key in samp.sims[0].keys():
        simPredicts[key] = np.empty([len(xs), len(ref.sims)])

    bootKeys = list(samp.sims[0].keys())
    for key in bootKeys:
        actualDiff[key] = [None for i in range(len(xs))]
        if not bkgd or key in ['K', 'k']:
            for i in range(len(xs)):
                if bkgd and xs[i] >= ref.bounds[0]:
                    actualDiff[key][i] = samp.interpSpectra[key](xs[i])
                elif xs[i] >= lower and xs[i] <= upper:
                    if diff == 'subtract':
                        if ref.interpSpectra[key](xs[i]) <= 0:
                            actualDiff[key][i] = samp.interpSpectra[key](xs[i])
                        else:
                            actualDiff[key][i] = samp.interpSpectra[key](xs[i]) - ref.interpSpectra[key](xs[i])
                    else:
                        if ref.interpSpectra[key](xs[i]) == 0:
                            actualDiff[key][i] = np.nan
                        else:
                            actualDiff[key][i] = samp.interpSpectra[key](xs[i]) / ref.interpSpectra[key](xs[i])
    
                for j in range(len(ref.sims)):
                    if bkgd and xs[i] >= ref.bounds[0]:
                        simPredicts[key][i,j] = samp.sims[j][key](xs[i])
                    elif xs[i] >= lower and xs[i] <= upper:
                        if diff == 'subtract':
                            if ref.sims[j][key](xs[i]) <= 0:
                                simPredicts[key][i,j] = samp.sims[j][key](xs[i])
                            else:
                                simPredicts[key][i,j] = samp.sims[j][key](xs[i]) - ref.sims[j][key](xs[i])
                        else:
                            if ref.sims[j][key](xs[i]) == 0:
                                simPredicts[key][i,j] = np.nan
                            else:
                                simPredicts[key][i,j] = samp.sims[j][key](xs[i]) / ref.sims[j][key](xs[i])
        
    if bkgd:
        for i in range(len(xs)):
            actualDiff['nm'][i] = actualDiff['K'][i]/samp.suswt
            actualDiff['ns'][i] = actualDiff['nm'][i]/(samp.BET*100**2)
            actualDiff['diffnm'][i] = actualDiff['k'][i]/samp.suswt
            actualDiff['diffns'][i] = actualDiff['diffnm'][i]/(samp.BET*100**2)
            
            simPredicts['nm'][i] = simPredicts['K'][i]/samp.suswt
            simPredicts['ns'][i] = simPredicts['nm'][i]/(samp.BET*100**2)
            simPredicts['diffnm'][i] = simPredicts['k'][i]/samp.suswt
            simPredicts['diffns'][i] = simPredicts['diffnm'][i]/(samp.BET*100**2)

    simPredicts = pd.DataFrame.from_dict({(key,j): simPredicts[key][:,j] for key in samp.sims[0].keys() for j in range(len(samp.sims))}, dtype = float)
    
    if CI == 'studentized':
        nresim = len(samp.resims[0])
        resimPredictsref = dict()
        resimPredictssamp = dict()
        for key in bootKeys:
            resimPredictsref[key] = dict()
            resimPredictssamp[key] = dict()
            for sim in range(nSim):
                resimPredictsref[key][sim] = dict()
                resimPredictssamp[key][sim] = dict()
                for i in range(len(xs)):
                    resimPredictsref[key][sim][i] = []
                    resimPredictssamp[key][sim][i] = []
                    for j in range(nresim):
                        resimPredictsref[key][sim][i].append(ref.resims[sim][j][key](xs[i]))
                        resimPredictssamp[key][sim][i].append(samp.resims[sim][j][key](xs[i]))
        resimPredictsref = pd.DataFrame.from_dict({(i,j,k): resimPredictsref[i][j][k] for i in resimPredictsref.keys() for j in resimPredictsref[i].keys() for k in resimPredictsref[i][j].keys()}, dtype=float)
        resimPredictssamp = pd.DataFrame.from_dict({(i,j,k): resimPredictssamp[i][j][k] for i in resimPredictssamp.keys() for j in resimPredictssamp[i].keys() for k in resimPredictssamp[i][j].keys()}, dtype=float)

    simMean = pd.DataFrame(np.empty([len(xs), len(samp.sims[0].keys())]), columns = samp.sims[0].keys())
    simStdev = pd.DataFrame(np.empty([len(xs), len(samp.sims[0].keys())]), columns = samp.sims[0].keys())
    simUpperCI = pd.DataFrame(np.empty([len(xs), len(samp.sims[0].keys())]), columns = samp.sims[0].keys())
    simLowerCI = pd.DataFrame(np.empty([len(xs), len(samp.sims[0].keys())]), columns = samp.sims[0].keys())
    if moreStats:
        simUpperQuantile = pd.DataFrame(np.empty([len(xs), len(samp.sims[0].keys())]), columns = samp.sims[0].keys())
        simLowerQuantile = pd.DataFrame(np.empty([len(xs), len(samp.sims[0].keys())]), columns = samp.sims[0].keys())
        
    for i in range(len(xs)):
        for key in bootKeys:
            samplemean = actualDiff[key][i]
            mean = simPredicts.loc[i, (key, slice(None))].mean()
            
            stdev = simPredicts.loc[i, (key, slice(None))].std()
                
            simMean[key][i] = mean
            simStdev[key][i] = stdev
            if moreStats:
                simUpperQuantile = simPredicts.loc[i, (key, slice(None))].quantile(q=alpha/2)
                simLowerQuantile = simPredicts.loc[i, (key, slice(None))].quantile(q=1-alpha/2)

            if CI == 'pivot':
                simLowerCI[key][i] = 2*samplemean - simPredicts.loc[i, (key, slice(None))].quantile(q=1-alpha/2)
                simUpperCI[key][i] = 2*samplemean - simPredicts.loc[i, (key, slice(None))].quantile(q=alpha/2)
            elif CI == 'quantile':
                simLowerCI[key][i] = simPredicts.loc[i, (key, slice(None))].quantile(q=alpha/2)
                simUpperCI[key][i] = simPredicts.loc[i, (key, slice(None))].quantile(q= 1 - alpha/2)
            elif CI == 'tboot':
                
                simLowerCI[key][i] = samplemean + t.ppf(alpha/2, n-1)*stdev
                simUpperCI[key][i] = samplemean + t.ppf(1-alpha/2, n-1)*stdev
            elif CI == 'tskew':
                kappa = simPredicts.loc[i,(key, slice(None))].skew(axis=0, skipna=True)/(6*np.sqrt(nSim))
                simLowerCI[key][i] = samplemean+(t.ppf(alpha/2, n-1) + kappa*(1 + 2*(t.ppf(alpha/2, n-1)**2)))*stdev
                simUpperCI[key][i] = samplemean+(t.ppf(1-alpha/2, n-1) + kappa*(1 + 2*(t.ppf(1-alpha/2, n-1)**2)))*stdev
            elif CI == 'expandedquantile':
                modalpha = norm.cdf(-np.sqrt(n/(n-1)) * -t.ppf(alpha/2, n-1))
                simLowerCI[key][i] = simPredicts.loc[i, (key, slice(None))].quantile(q=modalpha/2)
                simUpperCI[key][i] = simPredicts.loc[i, (key, slice(None))].quantile(q=1-modalpha/2)
            elif CI == 'studentized':
                resimstdev = []
                q = []
                for j in range(nSim):
                    if diff == 'subtract':
                        resimstdev.append(np.sqrt(resimPredictsref[key, j, i].std(axis=0, skipna=True)**2 + resimPredictssamp[key, j, i].std(axis=0, skipna=True)**2))
                    else:
                        if samp.sims[j][key](xs[i]) != 0 and ref.sims[j][key](xs[i]) != 0:
                            resimstdev.append(abs(simPredicts[key,j][i]*(np.sqrt((resimPredictsref[key, j, i].std(axis=0, skipna=True)/ref.sims[j][key](xs[i]))**2 + (resimPredictssamp[key, j, i].std(axis=0, skipna=True)/samp.sims[j][key](xs[i]))**2))))

                        elif samp.sims[j][key](xs[i]) == 0 and ref.sims[j][key](xs[i] == 0):
                            resimstdev.append(0)
                        elif samp.sims[j][key](xs[i]) == 0:
                            resimstdev.append(resimPredictssamp[key, j, i].std(axis=0, skipna=True))
                        else:
                            resimstdev.append(resimPredictsref[key, j, i].std(axis=0, skipna=True))

                    if resimstdev[j] == 0 or resimstdev[j] == np.inf:
                        if simPredicts[key, j][i] == mean or resimstdev[j] == np.inf:
                            q.append(0)
                        elif simPredicts[key, j][i] - mean < 0:
                            q.append(-np.inf)
                        else:
                            q.append(np.inf)
                    else:
                        q.append((simPredicts[key, j][i]-mean)/resimstdev[j])

                
                simLowerCI[key][i] = samplemean - stdev*np.quantile(q, 1-alpha/2)
                simUpperCI[key][i] = samplemean - stdev*np.quantile(q, alpha/2)
            
            #For debugging - helps to identify places where error bars 
            #cross the mean
            # if simLowerCI[key][i]>samplemean and key in ['k', 'K']:
            #     print('lower')
            #     print(simLowerCI[key][i], i)
            # if simUpperCI[key][i]<samplemean and key in ['k', 'K']:
            #     print('upper')
            #     print(simUpperCI[key][i], i)
            #     print('')
            #     print('Sample mean difference:')
            #     print(samplemean)
            #     print('Standard deviation')
            #     print(stdev)
            #     print('Simulations predicted:')
            #     print(simPredicts.loc[i, (key, slice(None))])
            #     print('')
            
    if diff == 'divide':
        simUpperCI[simUpperCI<0] = 0
        simLowerCI[simLowerCI<0] = 0
    
    actualDiff = pd.DataFrame(actualDiff)
        
    simLowerCI[simLowerCI>actualDiff] = actualDiff[simLowerCI>actualDiff]
    simUpperCI[simUpperCI<actualDiff] = actualDiff[simLowerCI<actualDiff]

    simMean.insert(0, 'Temp', xs)
    simStdev.insert(0, 'Temp', xs)
    simUpperCI.insert(0, 'Temp', xs)
    simLowerCI.insert(0, 'Temp', xs)
    if moreStats:
        simUpperQuantile.insert(0, 'Temp', xs)
        simLowerQuantile.insert(0, 'Temp', xs)
    interpStats = dict()
    interpStats['mean'] = interpolateWellDefined(simMean, modS=500)
    interpStats['stdev'] = interpolateWellDefined(simStdev, modS=500)
    interpStats['upperCI'] = interpolateWellDefined(simUpperCI, modS=500)
    interpStats['lowerCI'] = interpolateWellDefined(simLowerCI, modS=500)
    if moreStats:
        interpStats['upperQuantile'] = interpolateWellDefined(simUpperQuantile, modS=500)
        interpStats['lowerQuantile'] = interpolateWellDefined(simLowerQuantile, modS=500)
    actualDiff.insert(0, 'Temp', xs)
    actualDiff = interpolateWellDefined(actualDiff)
    
    #Uncomment to visualize CIs and interpolation of CIs 
    '''
    fig, where = plt.subplots(figsize=(12, 12))
    where.plot(xs, interpStats['upperCI']['k'](xs), ls='-', lw=3)
    where.plot(xs, interpStats['lowerCI']['k'](xs), ls='-', lw=3)
    where.plot(simUpperCI['Temp'], simUpperCI['k'], marker='o', lw=0)
    where.plot(simLowerCI['Temp'], simLowerCI['k'], marker='o', lw=0)
    where.set_yscale('log')
    '''
    
    #Background subtracts from all the simulations too
    if bkgd:
        for i in range(len(samp.sims)):
            temp = simPredicts.loc[slice(None), (slice(None), i)]
            temp.columns = temp.columns.droplevel(1)
            temp.insert(0, 'Temp', xs)
            sims[i] = interpolateSpectra(temp, interp=interp)
    bounds = (upper, lower)
    return actualDiff, interpStats, sims, bounds


#Easy function to load all of the freezes in a directory and all subdirectories
def loadAll(p = None, s = False):
    if p == None: p = input('Enter the path of your data folder: ')
    freezeDict = {}
    freezeDict = loadAllFunc(p, freezeDict, s)
    return freezeDict

def loadAllFunc(p, freezeDict, s):
    curr = None

    if not os.path.isdir(p):
        curr = loadFreeze(p, s=s)
        freezeDict[curr.name] = curr
    else:
        for filename in os.listdir(p):
            if filename != 'spectra':
                loadAllFunc(p + "/" + filename, freezeDict, s=s)
    return freezeDict


'''
Load a freeze from a CSV file saved using this program. 

If you set s to true, it will attempt to load the spectra you've saved with 
confidence intervals. 

'''
def loadFreeze(p, s=False):
    freeze = pd.read_csv(p)
    datatemp = freeze['data'].tolist()
    data = []
    for i in datatemp:
        if type(i) is not float or not np.isnan(i):
            data.append(i)
    att = set(freeze['att'].values)

    freeze = freeze[['Temp','dnF', 'nF']]


    freeze = freeze.apply(pd.to_numeric, errors='coerce')
    freeze.dropna(inplace=True)
    if p.rfind(os.sep) == -1:
        name = p[p.rfind('/')+1:-4]
    else:
        name = p[p.rfind(os.sep)+1:-4]
    '''
    if len(data) >= 7:
            bkgd=loadFreeze(data[-1], s=s)
    else: bkgd = None
    '''
    bkgd = None
    out = Freeze(freeze, data[:6], name, att, path=p, bkgd=bkgd)
    out.calcINAS()
    
    if s and (os.path.isdir(p[:p.rfind('/') + 1] + 'spectra') or os.path.isdir((p[:p.rfind(os.sep) + 1] + 'spectra'))):
        
        if p.rfind(os.sep) == -1:
            temp = pd.read_csv(p[:p.rfind('/') + 1] + 'spectra' + os.sep + name + '.csv')
        else:
            temp = pd.read_csv(p[:p.rfind(os.sep) + 1] + 'spectra' + os.sep + name + '.csv')
        temp = temp.apply(pd.to_numeric, errors='coerce')

        out.interpSpectra = dict()
        temps = temp['Temp (degrees Celsius)']
        interpSpectra = dict()
        interpSpectra['Temp'] = temps
        interpSpectra['FF'] = freeze['nF']/out.nD
        interpSpectra['ns'] = temp['ns'+ ' (Ice active sites per square centimeter)']
        interpSpectra['nm'] = interpSpectra['ns']*(out.BET*100**2)
        interpSpectra['K'] = interpSpectra['nm']*(out.suswt)
        interpSpectra['diffns'] = temp['diffns' + ' (Ice active sites per square centimeter degree Celsius)']
        interpSpectra['diffnm'] = interpSpectra['diffns']*(out.BET*100**2)
        interpSpectra['k'] = interpSpectra['diffnm']*out.suswt

        statstemp = dict()
        for i in ['lowerCI', 'upperCI']:
            statstemp[i] = dict()
            statstemp[i]['ns'] = temp['ns' + i + ' (Ice active sites per square centimeter)']
            statstemp[i]['nm'] = statstemp[i]['ns']*(out.BET*100**2)
            statstemp[i]['K'] = statstemp[i]['nm']*(out.suswt)
            statstemp[i]['diffns'] = temp['diffns' + i + ' (Ice active sites per square centimeter degree Celsius)']
            statstemp[i]['diffnm'] = statstemp[i]['diffns']*(out.BET*100**2)
            statstemp[i]['k'] = statstemp[i]['diffnm']*out.suswt
            statstemp[i]['Temp'] = temps

        out.interpSpectra = interpolateSpectra(pd.DataFrame.from_dict(interpSpectra))
        out.interpStats = dict()
        out.interpStats['lowerCI'] = interpolateWellDefined(pd.DataFrame.from_dict(statstemp['lowerCI']))
        out.interpStats['upperCI'] = interpolateWellDefined(pd.DataFrame.from_dict(statstemp['upperCI']))
        out.interpStats = pd.DataFrame.from_dict(out.interpStats)
        out.bounds = (temps[0], temps.iloc[-1])
    return out


#General utility function for calculating spectra and CBs for one function.
def calc(freeze, CI='tskew',interp='smoothedPCHIP', nSim = 1000):
    print('Interpolating and calculating CIs for %s' %freeze.name)
    freeze.calcINAS()
    freeze.interpolateSpectra(interp=interp)
    freeze.calcCI(nSim = nSim, CI=CI)


#Returns a list of difference experiments representing each member of lst
#compared to every other member of lst combined. 
def permComp(lst, CI='tskew', nSim = 1000):
    out = []
    
    for i in lst:
        calc(i, CI='tskew')
    
    for i in range(len(lst)):
        
        comb = []
        comb.extend(lst[:i])
        if i <len(lst):
            comb.extend(lst[i+1:])
        comb = combineLikeFreezes(comb, name=i)
        calc(comb)
        
        out.append(compareFreezes(lst[i], comb, CI='tskew'))

    
    return out