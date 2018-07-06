import pyfits 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import astropy 
from astropy import cosmology 
import math as mt
from scipy.integrate import quad
from scipy.stats import chi2_contingency
from pylab import *
from scipy.optimize import curve_fit
import scipy as sp
import scipy.special
import scipy.stats as stats
import seaborn as sns 
import pandas as pd
from astropy.modeling.models import Sersic1D
from numpy import * 
import bces.bces
import nmmn.stats

#https://github.com/rsnemmen/BCES/blob/master/stats%20howto%20bces.ipynb

#field data (_f)
f = open('/Users/jennifercooper/Documents/Gal_Ev/D4000_size/field_size_n.txt', 'r')
lines = f.readlines()[1:]
f.close()

#create arrays 
f_re_er = [] 
f_re    = []
f_n     = []
f_n_er  = []
f_z     = []
f_dk    = []
f_lmass = []
f_flag  = []
f_ans_re_1 =[]
f_dk_p  = []
f_dk_n = []
f_re_kpca = []
f_ssmd = []

#pull array column 
for line in lines: 
    a = line.split()
    f_z.append(float(a[16]))
    f_re.append(float(a[6]))
    f_re_er.append(float(a[7]))
    f_n.append(float(a[8]))
    f_n_er.append(float(a[9]))
    f_dk.append(float(a[32]))
    f_lmass.append(float(a[21]))
    f_flag.append(float(a[36]))
    f_dk_p.append(float(a[33]))
    f_dk_n.append(float(a[34]))
    f_re_kpca.append(float(a[37]))
    f_ssmd.append(float(a[38]))

#assign arrays 
f_re    = np.array(f_re)
f_re_er = np.array(f_re_er)
f_n     = np.array(f_n)
f_n_er  = np.array(f_n_er)
f_z     = np.array(f_z)
f_dk    = np.array(f_dk)
f_lmass = np.array(f_lmass)
f_flag  = np.array(f_flag) #greater than 0.9 (1) is SF, less than is Q
f_dk_p  = np.array(f_dk_p)
f_dk_n  = np.array(f_dk_n)
f_re_kpca = np.array(f_re_kpca)
f_dk_err = (f_dk_p + f_dk_n)/2. #average of errors 
f_ssmd  = np.array(f_ssmd)


f_re_kpc = f_re*8.615 #arc to kpc
f_b = 1.9992*f_n - 0.3271 #value of b


#one way to calculate the integrals 
#for indexb in range(len(f_b)):
#    for indexre in range(len(f_re_kpc)):
#        for indexn in range(len(f_n)):
#            def f1(x):
#                return 2*np.pi*x*np.exp(-f_b[indexb]*((x/f_re_kpc[indexre])**(1/f_n[indexn])-1))
#            
                  
#            a1, erra1 = quad(f1, 0, 1)
#            b1, errb1 = quad(f1, 0, np.inf)
#            f_ssmd = b1/a1
#            f_ssmd = np.array(f_ssmd)
#            print f_ssmd


#IRC 0218 data (_c)
c = open('/Users/jennifercooper/Documents/Gal_Ev/D4000_size/irc0218_size_n.txt', 'r')
lines = c.readlines()[1:]
c.close()

c_re_er = [] 
c_re    = []
c_n     = []
c_n_er  = []
c_z     = []
c_dk    = []
c_lmass = []
c_flag  = [] 
c_dk1   = []
c_dk2   = []
c_ssmd = []


for line in lines: 
    a = line.split()
    c_z.append(float(a[25]))
    c_re.append(float(a[6]))
    c_re_er.append(float(a[7]))
    c_n.append(float(a[8]))
    c_n_er.append(float(a[9]))
    c_dk.append(float(a[21]))
    c_lmass.append(float(a[30]))
    c_flag.append(float(a[36]))
    c_dk1.append(float(a[22]))
    c_dk2.append(float(a[23]))
    c_ssmd.append(float(a[37]))
    
c_re    = np.array(c_re)
c_re_er = np.array(c_re_er)
c_n     = np.array(c_n)
c_n_er  = np.array(c_n_er)
c_z     = np.array(c_z)
c_dk    = np.array(c_dk)
c_lmass = np.array(c_lmass)
c_flag  = np.array(c_flag)
c_dk1   = np.array(c_dk1)
c_dk2   = np.array(c_dk2)
c_ssmd  = np.array(c_ssmd)



c_re_kpc = c_re*8.615 #arc to kpc
c_b = 1.9992*c_n - 0.3271 #value of b


c_dk_err = (c_dk1 + c_dk2)/2.0 #average of errors on D4000

x = 0
y = 0
yer = 0
lcb1 = 0
lcb2 = 0
lcb3 = 0
ucb1 = 0
ucb2 = 0
ucb3 = 0
xcb1 = 0

import warnings
warnings.filterwarnings("ignore")

def fieldn(x,y,yer,lcb1,lcb2,lcb3,ucb1,ucb2,ucb3):
    ax=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
    ax = sns.set(style="white", color_codes=True)
    ax = subplot(2,2,1)
    ax = xlim(0.4,4)
    ax = ylim(-1,5)
    x, y, yer = pd.Series(f_n[np.where(f_flag>0.9)], name="Field Sersic Index SF"), pd.Series(f_dk[np.where(f_flag>0.9)], name="D4000"), pd.Series(f_dk_err[np.where(f_flag>0.9)], name="error")
    sort = np.argsort(x)
    x = x[sort]
    x = np.array(x)
    y = y[sort] 
    y = np.array(y)
    yer = yer[sort]
    yer = np.array(yer)
    xer=zeros(len(x))
    cov=zeros(len(x))   # no correlation between error measurements
    i = 0
    nboot=10000   # number of bootstrapping trials
    def func(x): return x[1]*x[0]+x[2]
    a,b,aerr,berr,covab=bces.bces.bcesp(x,xer,y,yer,cov,nboot)
    ybces=a[3]*x+b[3]  # the integer corresponds to the desired BCES method for plotting (3-ort, 0-y|x, 1-x|y, *don't use bissector*)
    # array with best-fit parameters
    fitm=np.array([ a[i],b[i] ])
    # covariance matrix of parameter uncertainties
    covm=np.array([ (aerr[i]**2,covab[i]), (covab[i],berr[i]**2) ])
    # Gets lower and upper bounds on the confidence band 
    lcb1,ucb1,x=nmmn.stats.confband(x, y, a[i], b[i], 0.68, x)
    lcb2,ucb2,x2=nmmn.stats.confband(x, y, a[i], b[i], 0.95, x)
    lcb3,ucb3,x3=nmmn.stats.confband(x, y, a[i], b[i], 0.997, x)
    errorbar(x,y,yerr=yer,fmt='o')
    ax = plot(x,a[i]*x+b[i],'-k')
    ax = fill_between(x, lcb1, ucb1, alpha=0.6, facecolor='purple')
    ax = fill_between(x, lcb2, ucb2, alpha=0.3, facecolor='blue')
    ax = fill_between(x, lcb3, ucb3, alpha=0.4, facecolor='grey')
    #ax = bar(x,y,yerr=[f_dk_p[np.where(f_flag>0.9)], f_dk_n[np.where(f_flag>0.9)]], facecolor='none')
    ax = legend(loc='best')
    ax = xlabel('Field Sersic Index SF')
    ax = ylabel('$D4000$')
    
    ax = subplot(2,2,2)
    ax = xlim(0.4,5)
    ax = ylim(-1,5)
    x, y = pd.Series(f_n[np.where(f_flag<0.9)], name="Field Sersic Index Q"), pd.Series(f_dk[np.where(f_flag<0.9)], name="D4000")
    yer = pd.Series(f_dk_err[np.where(f_flag<0.9)])
    sort = np.argsort(x)
    x = x[sort]
    x = np.array(x)
    y = y[sort] 
    y = np.array(y)
    yer = yer[sort]
    yer = np.array(yer)
    xer=zeros(len(x))
    cov=zeros(len(x))   # no correlation between error measurements
    i = 0
    nboot=10000   # number of bootstrapping trials
    def func(x): return x[1]*x[0]+x[2]
    a,b,aerr,berr,covab=bces.bces.bcesp(x,xer,y,yer,cov,nboot)
    ybces=a[3]*x+b[3]  # the integer corresponds to the desired BCES method for plotting (3-ort, 0-y|x, 1-x|y, *don't use bissector*)
    # array with best-fit parameters
    fitm=np.array([ a[i],b[i] ])
    # covariance matrix of parameter uncertainties
    covm=np.array([ (aerr[i]**2,covab[i]), (covab[i],berr[i]**2) ])
    # Gets lower and upper bounds on the confidence band 
    lcb1,ucb1,x=nmmn.stats.confband(x, y, a[i], b[i], 0.68, x)
    lcb2,ucb2,x2=nmmn.stats.confband(x, y, a[i], b[i], 0.95, x)
    lcb3,ucb3,x3=nmmn.stats.confband(x, y, a[i], b[i], 0.997, x)
    errorbar(x,y,yerr=yer,fmt='o')
    ax = plot(x,ybces,'-k')
    ax = fill_between(x, lcb1, ucb1, alpha=0.6, facecolor='purple')
    ax = fill_between(x, lcb2, ucb2, alpha=0.3, facecolor='blue')
    ax = fill_between(x, lcb3, ucb3, alpha=0.4, facecolor='grey')
    ax = xlabel('Field Sersic Index Q')
    purple_patch = mpatches.Patch(color='purple', label='68%')
    blue_patch = mpatches.Patch(color='blue', label='95%', alpha = 0.3)
    grey_patch = mpatches.Patch(color='grey', label='99.7%', alpha=0.4)
    plt.legend(handles=[purple_patch,blue_patch,grey_patch])
    
    
    
    ax = subplot(2,2,3)
    ax = xlim(0.4,4)
    ax = ylim(-1,5)
    x, y = pd.Series(f_n[np.where(np.logical_and(f_flag>0.9,f_lmass>10.20))], name="Field Sersic Index SF Mass Complete"), pd.Series(f_dk[np.where(np.logical_and(f_flag>0.9,f_lmass>10.20))], name="D4000")
    yer = pd.Series(f_dk_err[np.where(np.logical_and(f_flag>0.9,f_lmass>10.20))])
    sort = np.argsort(x)
    x = x[sort]
    x = np.array(x)
    y = y[sort] 
    y = np.array(y)
    yer = yer[sort]
    yer = np.array(yer)
    xer=zeros(len(x))
    cov=zeros(len(x))   # no correlation between error measurements
    i = 0
    nboot=10000   # number of bootstrapping trials
    def func(x): return x[1]*x[0]+x[2]
    a,b,aerr,berr,covab=bces.bces.bcesp(x,xer,y,yer,cov,nboot)
    ybces=a[3]*x+b[3]  # the integer corresponds to the desired BCES method for plotting (3-ort, 0-y|x, 1-x|y, *don't use bissector*)
    # array with best-fit parameters
    fitm=np.array([ a[i],b[i] ])
    # covariance matrix of parameter uncertainties
    covm=np.array([ (aerr[i]**2,covab[i]), (covab[i],berr[i]**2) ])
    # Gets lower and upper bounds on the confidence band 
    lcb1,ucb1,x=nmmn.stats.confband(x, y, a[i], b[i], 0.68, x)
    lcb2,ucb2,x2=nmmn.stats.confband(x, y, a[i], b[i], 0.95, x)
    lcb3,ucb3,x3=nmmn.stats.confband(x, y, a[i], b[i], 0.997, x)
    errorbar(x,y,yerr=yer,fmt='o')
    ax = plot(x,ybces,'-k')
    ax = fill_between(x, lcb1, ucb1, alpha=0.6, facecolor='purple')
    ax = fill_between(x, lcb2, ucb2, alpha=0.3, facecolor='blue')
    ax = fill_between(x, lcb3, ucb3, alpha=0.4, facecolor='grey')
    ax = xlabel('Field Sersic Index SF Mass Complete')
    ax = ylabel('D4000')
    ax = plt.show()
    
def ircn(x,y,yer,lcb1,lcb2,lcb3,ucb1,ucb2,ucb3):
    ax=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
    ax = sns.set(style="white", color_codes=True)
    ax = subplot(2,2,1)
    ax = ylim(0,2.5)
    ax = xlabel('IRC0218 Sersic Index Q')
    ax = ylabel('D4000')
    x, y = pd.Series(c_n[np.where(c_flag<0.9)], name="IRC0218 Sersic Index Q"), pd.Series(c_dk[np.where(c_flag<0.9)], name="D4000")
    yer = c_dk_err[np.where(c_flag<0.9)]
    sort = np.argsort(x)
    x = x[sort]
    x = np.array(x)
    y = y[sort] 
    y = np.array(y)
    yer = yer[sort]
    yer = np.array(yer)
    xer=zeros(len(x))
    cov=zeros(len(x))   # no correlation between error measurements
    i = 0
    nboot=10000   # number of bootstrapping trials
    def func(x): return x[1]*x[0]+x[2]
    a,b,aerr,berr,covab=bces.bces.bcesp(x,xer,y,yer,cov,nboot)
    ybces=a[3]*x+b[3]  # the integer corresponds to the desired BCES method for plotting (3-ort, 0-y|x, 1-x|y, *don't use bissector*)
    # array with best-fit parameters
    fitm=np.array([ a[i],b[i] ])
    # covariance matrix of parameter uncertainties
    covm=np.array([ (aerr[i]**2,covab[i]), (covab[i],berr[i]**2) ])
    # Gets lower and upper bounds on the confidence band 
    lcb1,ucb1,x=nmmn.stats.confband(x, y, a[i], b[i], 0.68, x)
    lcb2,ucb2,x2=nmmn.stats.confband(x, y, a[i], b[i], 0.95, x)
    lcb3,ucb3,x3=nmmn.stats.confband(x, y, a[i], b[i], 0.997, x)
    errorbar(x,y,yerr=yer,fmt='o')
    ax = plot(x,ybces,'-k')
    ax = fill_between(x, lcb1, ucb1, alpha=0.6, facecolor='purple')
    ax = fill_between(x, lcb2, ucb2, alpha=0.3, facecolor='blue')
    ax = fill_between(x, lcb3, ucb3, alpha=0.4, facecolor='grey')
    
    ax = subplot(2,2,2)
    ax = ylim(0.5,2)
    ax = xlabel('IRC0218 Sersic Index SF')
    ax = ylabel('D4000')
    x, y = pd.Series(c_n[np.where(c_flag>0.9)], name="IRC0218 Sersic Index SF"), pd.Series(c_dk[np.where(c_flag>0.9)], name="D4000")
    yer = c_dk_err[np.where(c_flag>0.9)]
    sort = np.argsort(x)
    x = x[sort]
    x = np.array(x)
    y = y[sort] 
    y = np.array(y)
    yer = yer[sort]
    yer = np.array(yer)
    xer=zeros(len(x))
    cov=zeros(len(x))   # no correlation between error measurements
    i = 0
    nboot=10000   # number of bootstrapping trials
    def func(x): return x[1]*x[0]+x[2]
    a,b,aerr,berr,covab=bces.bces.bcesp(x,xer,y,yer,cov,nboot)
    ybces=a[3]*x+b[3]  # the integer corresponds to the desired BCES method for plotting (3-ort, 0-y|x, 1-x|y, *don't use bissector*)
    # array with best-fit parameters
    fitm=np.array([ a[i],b[i] ])
    # covariance matrix of parameter uncertainties
    covm=np.array([ (aerr[i]**2,covab[i]), (covab[i],berr[i]**2) ])
    # Gets lower and upper bounds on the confidence band 
    lcb1,ucb1,x=nmmn.stats.confband(x, y, a[i], b[i], 0.68, x)
    lcb2,ucb2,x2=nmmn.stats.confband(x, y, a[i], b[i], 0.95, x)
    lcb3,ucb3,x3=nmmn.stats.confband(x, y, a[i], b[i], 0.997, x)
    errorbar(x,y,yerr=yer,fmt='o')
    ax = plot(x,ybces,'-k')    
    ax = fill_between(x, lcb1, ucb1, alpha=0.6, facecolor='purple')
    ax = fill_between(x, lcb2, ucb2, alpha=0.3, facecolor='blue')
    ax = fill_between(x, lcb3, ucb3, alpha=0.4, facecolor='grey')
    purple_patch = mpatches.Patch(color='purple', label='68%')
    blue_patch = mpatches.Patch(color='blue', label='95%', alpha = 0.3)
    grey_patch = mpatches.Patch(color='grey', label='99.7%', alpha=0.4)
    plt.legend(handles=[purple_patch,blue_patch,grey_patch])
    
    ax = subplot(2,2,3)
    ax = ylim(0,2.5)
    ax = ylabel('D4000')
    ax = xlabel('IRC0218 Sersic Index Q Mass Complete')
    x, y = pd.Series(c_n[np.where(np.logical_and(c_flag<0.9,c_lmass>10.20))], name="IRC0218 Sersic Index Q Mass Complete"), pd.Series(c_dk[np.where(np.logical_and(c_flag<0.9,c_lmass>10.20))], name="D4000")
    yer = c_dk_err[np.where(np.logical_and(c_flag<0.9,c_lmass>10.20))]
    sort = np.argsort(x)
    x = x[sort]
    x = np.array(x)
    y = y[sort] 
    y = np.array(y)
    yer = yer[sort]
    yer = np.array(yer)
    xer=zeros(len(x))
    cov=zeros(len(x))   # no correlation between error measurements
    i = 0
    nboot=10000   # number of bootstrapping trials
    def func(x): return x[1]*x[0]+x[2]
    a,b,aerr,berr,covab=bces.bces.bcesp(x,xer,y,yer,cov,nboot)
    ybces=a[3]*x+b[3]  # the integer corresponds to the desired BCES method for plotting (3-ort, 0-y|x, 1-x|y, *don't use bissector*)
    # array with best-fit parameters
    fitm=np.array([ a[i],b[i] ])
    # covariance matrix of parameter uncertainties
    covm=np.array([ (aerr[i]**2,covab[i]), (covab[i],berr[i]**2) ])
    # Gets lower and upper bounds on the confidence band 
    lcb1,ucb1,x=nmmn.stats.confband(x, y, a[i], b[i], 0.68, x)
    lcb2,ucb2,x2=nmmn.stats.confband(x, y, a[i], b[i], 0.95, x)
    lcb3,ucb3,x3=nmmn.stats.confband(x, y, a[i], b[i], 0.997, x)
    errorbar(x,y,yerr=yer,fmt='o')
    ax = plot(x,ybces,'-k')
    ax = fill_between(x, lcb1, ucb1, alpha=0.6, facecolor='purple')
    ax = fill_between(x, lcb2, ucb2, alpha=0.3, facecolor='blue')
    ax = fill_between(x, lcb3, ucb3, alpha=0.4, facecolor='grey')
    
    
    ax = subplot(2,2,4)
    ax = ylim(0,2.5)
    ax = xlabel('IRC0218 Sersic Index SF Mass Complete')
    ax = ylabel('D4000')
    x = c_n[np.where(np.logical_and(c_flag>0.9,c_lmass>10.20))]
    y = c_dk[np.where(np.logical_and(c_flag>0.9,c_lmass>10.20))]
    yer = c_dk_err[np.where(np.logical_and(c_flag>0.9,c_lmass>10.20))]
    sort = np.argsort(x)
    x = x[sort]
    x = np.array(x)
    y = y[sort] 
    y = np.array(y)
    yer = yer[sort]
    yer = np.array(yer)
    xer=zeros(len(x))
    cov=zeros(len(x))   # no correlation between error measurements
    i = 0
    nboot=10000   # number of bootstrapping trials
    def func(x): return x[1]*x[0]+x[2]
    a,b,aerr,berr,covab=bces.bces.bcesp(x,xer,y,yer,cov,nboot)
    ybces=a[3]*x+b[3]  # the integer corresponds to the desired BCES method for plotting (3-ort, 0-y|x, 1-x|y, *don't use bissector*)
    # array with best-fit parameters
    fitm=np.array([ a[i],b[i] ])
    # covariance matrix of parameter uncertainties
    covm=np.array([ (aerr[i]**2,covab[i]), (covab[i],berr[i]**2) ])
    # Gets lower and upper bounds on the confidence band 
    lcb1,ucb1,x=nmmn.stats.confband(x, y, a[i], b[i], 0.68, x)
    lcb2,ucb2,x2=nmmn.stats.confband(x, y, a[i], b[i], 0.95, x)
    lcb3,ucb3,x3=nmmn.stats.confband(x, y, a[i], b[i], 0.997, x)
    errorbar(x,y,yerr=yer,fmt='o')
    ybces = np.array([ 1.25897495,1.2628083,1.26300836,1.30052192,1.30430883,1.31449953,1.32968822])
    ax = plot(x,a[i]*x+b[i],'-k')
    ax = fill_between(x, lcb1, ucb1, alpha=0.6, facecolor='purple')
    ax = fill_between(x, lcb2, ucb2, alpha=0.3, facecolor='blue')
    ax = fill_between(x, lcb3, ucb3, alpha=0.4, facecolor='grey')
    plt.show()

def fieldr(x,y,yer,lcb1,lcb2,lcb3,ucb1,ucb2,ucb3):
    ax=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
    ax = sns.set(style="white", color_codes=True)
    ax = subplot(2,2,1)
    ax = ylim(0,5)
    x, y = pd.Series(f_re_kpc[np.where(f_flag>0.9)]), pd.Series(f_dk[np.where(f_flag>0.9)])
    yer = f_dk_err[np.where(f_flag>0.9)]
    sort = np.argsort(x)
    x = x[sort]
    x = np.array(x)
    y = y[sort] 
    y = np.array(y)
    yer = yer[sort]
    yer = np.array(yer)
    xer=zeros(len(x))
    cov=zeros(len(x))   # no correlation between error measurements
    i = 0
    nboot=10000   # number of bootstrapping trials
    def func(x): return x[1]*x[0]+x[2]
    a,b,aerr,berr,covab=bces.bces.bcesp(x,xer,y,yer,cov,nboot)
    ybces=a[3]*x+b[3]  # the integer corresponds to the desired BCES method for plotting (3-ort, 0-y|x, 1-x|y, *don't use bissector*)
    # array with best-fit parameters
    fitm=np.array([ a[i],b[i] ])
    # covariance matrix of parameter uncertainties
    covm=np.array([ (aerr[i]**2,covab[i]), (covab[i],berr[i]**2) ])
    # Gets lower and upper bounds on the confidence band 
    lcb1,ucb1,x=nmmn.stats.confband(x, y, a[i], b[i], 0.68, x)
    lcb2,ucb2,x2=nmmn.stats.confband(x, y, a[i], b[i], 0.95, x)
    lcb3,ucb3,x3=nmmn.stats.confband(x, y, a[i], b[i], 0.997, x)
    errorbar(x,y,yerr=yer,fmt='o')
    ax = plot(x,a[i]*x+b[i],'-k')
    ax = fill_between(x, lcb1, ucb1, alpha=0.6, facecolor='purple')
    ax = fill_between(x, lcb2, ucb2, alpha=0.3, facecolor='blue')
    ax = fill_between(x, lcb3, ucb3, alpha=0.4, facecolor='grey')
    #ax = bar(x,y,yerr=[f_dk_p[np.where(f_flag>0.9)], f_dk_n[np.where(f_flag>0.9)]], facecolor='none')
    ax = legend(loc='best')
    ax = xlabel('Field Radius Effective kpc SF')
    ax = ylabel('$D4000$')
    
    ax = subplot(2,2,2)
   # ax = xlim(0.4,5)
   # ax = ylim(-1,5)
    x, y = pd.Series(f_re_kpc[np.where(f_flag<0.9)]), pd.Series(f_dk[np.where(f_flag<0.9)])
    yer = pd.Series(f_dk_err[np.where(f_flag<0.9)])
    sort = np.argsort(x)
    x = x[sort]
    x = np.array(x)
    y = y[sort] 
    y = np.array(y)
    yer = yer[sort]
    yer = np.array(yer)
    xer=zeros(len(x))
    cov=zeros(len(x))   # no correlation between error measurements
    i = 0
    nboot=10000   # number of bootstrapping trials
    def func(x): return x[1]*x[0]+x[2]
    a,b,aerr,berr,covab=bces.bces.bcesp(x,xer,y,yer,cov,nboot)
    ybces=a[3]*x+b[3]  # the integer corresponds to the desired BCES method for plotting (3-ort, 0-y|x, 1-x|y, *don't use bissector*)
    # array with best-fit parameters
    fitm=np.array([ a[i],b[i] ])
    # covariance matrix of parameter uncertainties
    covm=np.array([ (aerr[i]**2,covab[i]), (covab[i],berr[i]**2) ])
    # Gets lower and upper bounds on the confidence band 
    lcb1,ucb1,x=nmmn.stats.confband(x, y, a[i], b[i], 0.68, x)
    lcb2,ucb2,x2=nmmn.stats.confband(x, y, a[i], b[i], 0.95, x)
    lcb3,ucb3,x3=nmmn.stats.confband(x, y, a[i], b[i], 0.997, x)
    errorbar(x,y,yerr=yer,fmt='o')
    ax = plot(x,ybces,'-k')
    ax = fill_between(x, lcb1, ucb1, alpha=0.6, facecolor='purple')
    ax = fill_between(x, lcb2, ucb2, alpha=0.3, facecolor='blue')
    ax = fill_between(x, lcb3, ucb3, alpha=0.4, facecolor='grey')
    ax = xlabel('Field Radius Effective kpc Q')
    purple_patch = mpatches.Patch(color='purple', label='68%')
    blue_patch = mpatches.Patch(color='blue', label='95%', alpha = 0.3)
    grey_patch = mpatches.Patch(color='grey', label='99.7%', alpha=0.4)
    plt.legend(handles=[purple_patch,blue_patch,grey_patch])
    
    
    
    ax = subplot(2,2,3)
  #  ax = xlim(0.4,4)
  #  ax = ylim(-1,5)
    x, y = pd.Series(f_re_kpc[np.where(np.logical_and(f_flag>0.9,f_lmass>10.20))]), pd.Series(f_dk[np.where(np.logical_and(f_flag>0.9,f_lmass>10.20))])
    yer = pd.Series(f_dk_err[np.where(np.logical_and(f_flag>0.9,f_lmass>10.20))])
    sort = np.argsort(x)
    x = x[sort]
    x = np.array(x)
    y = y[sort] 
    y = np.array(y)
    yer = yer[sort]
    yer = np.array(yer)
    xer=zeros(len(x))
    cov=zeros(len(x))   # no correlation between error measurements
    i = 0
    nboot=10000   # number of bootstrapping trials
    def func(x): return x[1]*x[0]+x[2]
    a,b,aerr,berr,covab=bces.bces.bcesp(x,xer,y,yer,cov,nboot)
    ybces=a[3]*x+b[3]  # the integer corresponds to the desired BCES method for plotting (3-ort, 0-y|x, 1-x|y, *don't use bissector*)
    # array with best-fit parameters
    fitm=np.array([ a[i],b[i] ])
    # covariance matrix of parameter uncertainties
    covm=np.array([ (aerr[i]**2,covab[i]), (covab[i],berr[i]**2) ])
    # Gets lower and upper bounds on the confidence band 
    lcb1,ucb1,x=nmmn.stats.confband(x, y, a[i], b[i], 0.68, x)
    lcb2,ucb2,x2=nmmn.stats.confband(x, y, a[i], b[i], 0.95, x)
    lcb3,ucb3,x3=nmmn.stats.confband(x, y, a[i], b[i], 0.997, x)
    errorbar(x,y,yerr=yer,fmt='o')
    ax = plot(x,ybces,'-k')
    ax = fill_between(x, lcb1, ucb1, alpha=0.6, facecolor='purple')
    ax = fill_between(x, lcb2, ucb2, alpha=0.3, facecolor='blue')
    ax = fill_between(x, lcb3, ucb3, alpha=0.4, facecolor='grey')
    ax = xlabel('Field Radius Effective kpc SF Mass Complete')
    ax = ylabel('D4000')
    ax = plt.show()



def ircr(x,y,yer,lcb1,lcb2,lcb3,ucb1,ucb2,ucb3):
    ax = sns.set(style="white", color_codes=True)
    ax = plt.figure(figsize=(18, 16), dpi= 80, facecolor='none', edgecolor='k')
    ax = subplot(2,2,1)
    ax = ylim(0,2.5)
    x, y = pd.Series(c_re_kpc[np.where(c_flag<0.9)], name="IRC0218 Radius kpc Q"), pd.Series(c_dk[np.where(c_flag<0.9)], name="D4000")
    yer = f_dk_err[np.where(c_flag<0.9)]
    sort = np.argsort(x)
    x = x[sort]
    x = np.array(x)
    y = y[sort] 
    y = np.array(y)
    yer = yer[sort]
    yer = np.array(yer)
    xer=zeros(len(x))
    cov=zeros(len(x))   # no correlation between error measurements
    i = 0
    nboot=10000   # number of bootstrapping trials
    def func(x): return x[1]*x[0]+x[2]
    a,b,aerr,berr,covab=bces.bces.bcesp(x,xer,y,yer,cov,nboot)
    ybces=a[3]*x+b[3]  # the integer corresponds to the desired BCES method for plotting (3-ort, 0-y|x, 1-x|y, *don't use bissector*)
    # array with best-fit parameters
    fitm=np.array([ a[i],b[i] ])
    # covariance matrix of parameter uncertainties
    covm=np.array([ (aerr[i]**2,covab[i]), (covab[i],berr[i]**2) ])
    # Gets lower and upper bounds on the confidence band 
    lcb1,ucb1,x=nmmn.stats.confband(x, y, a[i], b[i], 0.68, x)
    lcb2,ucb2,x2=nmmn.stats.confband(x, y, a[i], b[i], 0.95, x)
    lcb3,ucb3,x3=nmmn.stats.confband(x, y, a[i], b[i], 0.997, x)
    errorbar(x,y,yerr=yer,fmt='o')
    ax = plot(x,ybces,'-k')
    ax = fill_between(x, lcb1, ucb1, alpha=0.6, facecolor='purple')
    ax = fill_between(x, lcb2, ucb2, alpha=0.3, facecolor='blue')
    ax = fill_between(x, lcb3, ucb3, alpha=0.4, facecolor='grey')
    ax = xlabel('IRC 0218 Radius kpc Q')
    ax = ylabel('D4000')
    
    ax = subplot(2,2,2)
    ax = ylim(0,2.5)
    x, y = pd.Series(c_re_kpc[np.where(c_flag>0.9)], name="IRC0218 Radius kpc SF"), pd.Series(c_dk[np.where(c_flag>0.9)], name="D4000")
    yer = f_dk_err[np.where(c_flag>0.9)]
    sort = np.argsort(x)
    x = x[sort]
    x = np.array(x)
    y = y[sort] 
    y = np.array(y)
    yer = yer[sort]
    yer = np.array(yer)
    xer=zeros(len(x))
    cov=zeros(len(x))   # no correlation between error measurements
    i = 0
    nboot=10000   # number of bootstrapping trials
    def func(x): return x[1]*x[0]+x[2]
    a,b,aerr,berr,covab=bces.bces.bcesp(x,xer,y,yer,cov,nboot)
    ybces=a[3]*x+b[3]  # the integer corresponds to the desired BCES method for plotting (3-ort, 0-y|x, 1-x|y, *don't use bissector*)
    # array with best-fit parameters
    fitm=np.array([ a[i],b[i] ])
    # covariance matrix of parameter uncertainties
    covm=np.array([ (aerr[i]**2,covab[i]), (covab[i],berr[i]**2) ])
    # Gets lower and upper bounds on the confidence band 
    lcb1,ucb1,x=nmmn.stats.confband(x, y, a[i], b[i], 0.68, x)
    lcb2,ucb2,x2=nmmn.stats.confband(x, y, a[i], b[i], 0.95, x)
    lcb3,ucb3,x3=nmmn.stats.confband(x, y, a[i], b[i], 0.997, x)
    errorbar(x,y,yerr=yer,fmt='o')
    ax = plot(x,ybces,'-k')
    ax = fill_between(x, lcb1, ucb1, alpha=0.6, facecolor='purple')
    ax = fill_between(x, lcb2, ucb2, alpha=0.3, facecolor='blue')
    ax = fill_between(x, lcb3, ucb3, alpha=0.4, facecolor='grey')
    ax = xlabel('IRC Radius kpc SF')
    ax = ylabel('D4000')
    purple_patch = mpatches.Patch(color='purple', label='68%')
    blue_patch = mpatches.Patch(color='blue', label='95%', alpha = 0.3)
    grey_patch = mpatches.Patch(color='grey', label='99.7%', alpha=0.4)
    plt.legend(handles=[purple_patch,blue_patch,grey_patch])
    
    
    ax = subplot(2,2,3)
    ax = ylim(0,2.5)
    x, y = pd.Series(c_re_kpc[np.where(np.logical_and(c_flag<0.9,c_lmass>10.20))], name="IRC0218 Radius kpc Q Mass Complete"), pd.Series(c_dk[np.where(np.logical_and(c_flag<0.9,c_lmass>10.20))], name="D4000")
    yer = f_dk_err[np.where(np.logical_and(c_flag<0.9,c_lmass>10.20))]
    sort = np.argsort(x)
    x = x[sort]
    x = np.array(x)
    y = y[sort] 
    y = np.array(y)
    yer = yer[sort]
    yer = np.array(yer)
    xer=zeros(len(x))
    cov=zeros(len(x))   # no correlation between error measurements
    i = 0
    nboot=10000   # number of bootstrapping trials
    def func(x): return x[1]*x[0]+x[2]
    a,b,aerr,berr,covab=bces.bces.bcesp(x,xer,y,yer,cov,nboot)
    ybces=a[3]*x+b[3]  # the integer corresponds to the desired BCES method for plotting (3-ort, 0-y|x, 1-x|y, *don't use bissector*)
    # array with best-fit parameters
    fitm=np.array([ a[i],b[i] ])
    # covariance matrix of parameter uncertainties
    covm=np.array([ (aerr[i]**2,covab[i]), (covab[i],berr[i]**2) ])
    # Gets lower and upper bounds on the confidence band 
    lcb1,ucb1,x=nmmn.stats.confband(x, y, a[i], b[i], 0.68, x)
    lcb2,ucb2,x2=nmmn.stats.confband(x, y, a[i], b[i], 0.95, x)
    lcb3,ucb3,x3=nmmn.stats.confband(x, y, a[i], b[i], 0.997, x)
    errorbar(x,y,yerr=yer,fmt='o')
    ax = plot(x,ybces,'-k')
    ax = fill_between(x, lcb1, ucb1, alpha=0.6, facecolor='purple')
    ax = fill_between(x, lcb2, ucb2, alpha=0.3, facecolor='blue')
    ax = fill_between(x, lcb3, ucb3, alpha=0.4, facecolor='grey')
    ax = xlabel('IRC 0218 Radius kpc Q Mass Complete')
    ax = ylabel('D4000')
    
    
    ax = subplot(2,2,4)
    ax = ylim(0,2.5)
    x, y = pd.Series(c_re_kpc[np.where(np.logical_and(c_flag>0.9,c_lmass>10.20))], name="IRC0218 Radius kpc SF Mass Complete"), pd.Series(c_dk[np.where(np.logical_and(c_flag>0.9,c_lmass>10.20))], name="D4000")
    yer = c_dk_err[np.where(np.logical_and(c_flag>0.9,c_lmass>10.20))]
    sort = np.argsort(x)
    x = x[sort]
    x = np.array(x)
    y = y[sort] 
    y = np.array(y)
    yer = yer[sort]
    yer = np.array(yer)
    xer=zeros(len(x))
    cov=zeros(len(x))   # no correlation between error measurements
    i = 0
    nboot=10000   # number of bootstrapping trials
    def func(x): return x[1]*x[0]+x[2]
    a,b,aerr,berr,covab=bces.bces.bcesp(x,xer,y,yer,cov,nboot)
    ybces=a[3]*x+b[3]  # the integer corresponds to the desired BCES method for plotting (3-ort, 0-y|x, 1-x|y, *don't use bissector*)
    # array with best-fit parameters
    fitm=np.array([ a[i],b[i] ])
    # covariance matrix of parameter uncertainties
    covm=np.array([ (aerr[i]**2,covab[i]), (covab[i],berr[i]**2) ])
    # Gets lower and upper bounds on the confidence band 
    lcb1,ucb1,x=nmmn.stats.confband(x, y, a[i], b[i], 0.68, x)
    lcb2,ucb2,x2=nmmn.stats.confband(x, y, a[i], b[i], 0.95, x)
    lcb3,ucb3,x3=nmmn.stats.confband(x, y, a[i], b[i], 0.997, x)
    errorbar(x,y,yerr=yer,fmt='o')
    ax = plot(x,a[i]*x+b[i],'-k')
    ax = fill_between(x, lcb1, ucb1, alpha=0.6, facecolor='purple')
    ax = fill_between(x, lcb2, ucb2, alpha=0.3, facecolor='blue')
    ax = fill_between(x, lcb3, ucb3, alpha=0.4, facecolor='grey')
    ax = xlabel('IRC 0218 Radius kpc SF Mass Complete')
    ax = ylabel('D4000')
    plt.show()

def fieldssmd(x,y,yer,lcb1,lcb2,lcb3,ucb1,ucb2,ucb3):
    ax=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
    ax = sns.set(style="white", color_codes=True)
    ax = subplot(2,2,1)
    #ax = xlim(0.4,4)
    ax = ylim(-3,5)
    x, y, yer = pd.Series(f_ssmd[np.where(f_flag>0.9)], name="Field Sersic Index SF"), pd.Series(f_dk[np.where(f_flag>0.9)], name="D4000"), pd.Series(f_dk_err[np.where(f_flag>0.9)], name="error")
    sort = np.argsort(x)
    x = x[sort]
    x = np.array(x)
    y = y[sort] 
    y = np.array(y)
    yer = yer[sort]
    yer = np.array(yer)
    xer=zeros(len(x))
    cov=zeros(len(x))   # no correlation between error measurements
    i = 0
    nboot=10000   # number of bootstrapping trials
    def func(x): return x[1]*x[0]+x[2]
    a,b,aerr,berr,covab=bces.bces.bcesp(x,xer,y,yer,cov,nboot)
    ybces=a[3]*x+b[3]  # the integer corresponds to the desired BCES method for plotting (3-ort, 0-y|x, 1-x|y, *don't use bissector*)
    # array with best-fit parameters
    fitm=np.array([ a[i],b[i] ])
    # covariance matrix of parameter uncertainties
    covm=np.array([ (aerr[i]**2,covab[i]), (covab[i],berr[i]**2) ])
    # Gets lower and upper bounds on the confidence band 
    lcb1,ucb1,x=nmmn.stats.confband(x, y, a[i], b[i], 0.68, x)
    lcb2,ucb2,x2=nmmn.stats.confband(x, y, a[i], b[i], 0.95, x)
    lcb3,ucb3,x3=nmmn.stats.confband(x, y, a[i], b[i], 0.997, x)
    errorbar(x,y,yerr=yer,fmt='o')
    ax = plot(x,ybces,'-k')
    ax = fill_between(x, lcb1, ucb1, alpha=0.6, facecolor='purple')
    ax = fill_between(x, lcb2, ucb2, alpha=0.3, facecolor='blue')
    ax = fill_between(x, lcb3, ucb3, alpha=0.4, facecolor='grey')
    #ax = bar(x,y,yerr=[f_dk_p[np.where(f_flag>0.9)], f_dk_n[np.where(f_flag>0.9)]], facecolor='none')
    ax = legend(loc='best')
    ax = xlabel('Field $\Sigma_{1kpc}$ SF')
    ax = ylabel('$D4000$')
    
    ax = subplot(2,2,2)
    #ax = xlim(0.4,5)
    #ax = ylim(-1,5)
    x, y = pd.Series(f_ssmd[np.where(f_flag<0.9)], name="Field Sersic Index Q"), pd.Series(f_dk[np.where(f_flag<0.9)], name="D4000")
    yer = pd.Series(f_dk_err[np.where(f_flag<0.9)])
    sort = np.argsort(x)
    x = x[sort]
    x = np.array(x)
    y = y[sort] 
    y = np.array(y)
    yer = yer[sort]
    yer = np.array(yer)
    xer=zeros(len(x))
    cov=zeros(len(x))   # no correlation between error measurements
    i = 0
    nboot=10000   # number of bootstrapping trials
    def func(x): return x[1]*x[0]+x[2]
    a,b,aerr,berr,covab=bces.bces.bcesp(x,xer,y,yer,cov,nboot)
    ybces=a[3]*x+b[3]  # the integer corresponds to the desired BCES method for plotting (3-ort, 0-y|x, 1-x|y, *don't use bissector*)
    # array with best-fit parameters
    fitm=np.array([ a[i],b[i] ])
    # covariance matrix of parameter uncertainties
    covm=np.array([ (aerr[i]**2,covab[i]), (covab[i],berr[i]**2) ])
    # Gets lower and upper bounds on the confidence band 
    lcb1,ucb1,x=nmmn.stats.confband(x, y, a[i], b[i], 0.68, x)
    lcb2,ucb2,x2=nmmn.stats.confband(x, y, a[i], b[i], 0.95, x)
    lcb3,ucb3,x3=nmmn.stats.confband(x, y, a[i], b[i], 0.997, x)
    errorbar(x,y,yerr=yer,fmt='o')
    ax = plot(x,ybces,'-k')
    ax = fill_between(x, lcb1, ucb1, alpha=0.6, facecolor='purple')
    ax = fill_between(x, lcb2, ucb2, alpha=0.3, facecolor='blue')
    ax = fill_between(x, lcb3, ucb3, alpha=0.4, facecolor='grey')
    ax = xlabel('Field $\Sigma_{1kpc}$ Q')
    purple_patch = mpatches.Patch(color='purple', label='68%')
    blue_patch = mpatches.Patch(color='blue', label='95%', alpha = 0.3)
    grey_patch = mpatches.Patch(color='grey', label='99.7%', alpha=0.4)
    plt.legend(handles=[purple_patch,blue_patch,grey_patch])
    
    
    
    ax = subplot(2,2,3)
    #ax = xlim(0.4,4)
    #ax = ylim(-1,5)
    x, y = pd.Series(f_ssmd[np.where(np.logical_and(f_flag>0.9,f_lmass>10.20))], name="Field Sersic Index SF Mass Complete"), pd.Series(f_dk[np.where(np.logical_and(f_flag>0.9,f_lmass>10.20))], name="D4000")
    yer = pd.Series(f_dk_err[np.where(np.logical_and(f_flag>0.9,f_lmass>10.20))])
    sort = np.argsort(x)
    x = x[sort]
    x = np.array(x)
    y = y[sort] 
    y = np.array(y)
    yer = yer[sort]
    yer = np.array(yer)
    xer=zeros(len(x))
    cov=zeros(len(x))   # no correlation between error measurements
    i = 0
    nboot=10000   # number of bootstrapping trials
    def func(x): return x[1]*x[0]+x[2]
    a,b,aerr,berr,covab=bces.bces.bcesp(x,xer,y,yer,cov,nboot)
    ybces=a[3]*x+b[3]  # the integer corresponds to the desired BCES method for plotting (3-ort, 0-y|x, 1-x|y, *don't use bissector*)
    # array with best-fit parameters
    fitm=np.array([ a[i],b[i] ])
    # covariance matrix of parameter uncertainties
    covm=np.array([ (aerr[i]**2,covab[i]), (covab[i],berr[i]**2) ])
    # Gets lower and upper bounds on the confidence band 
    lcb1,ucb1,x=nmmn.stats.confband(x, y, a[i], b[i], 0.68, x)
    lcb2,ucb2,x2=nmmn.stats.confband(x, y, a[i], b[i], 0.95, x)
    lcb3,ucb3,x3=nmmn.stats.confband(x, y, a[i], b[i], 0.997, x)
    errorbar(x,y,yerr=yer,fmt='o')
    ax = plot(x,ybces,'-k')
    ax = fill_between(x, lcb1, ucb1, alpha=0.6, facecolor='purple')
    ax = fill_between(x, lcb2, ucb2, alpha=0.3, facecolor='blue')
    ax = fill_between(x, lcb3, ucb3, alpha=0.4, facecolor='grey')
    ax = xlabel('Field $\Sigma_{1kpc}$ SF Mass Complete')
    ax = ylabel('D4000')
    ax = plt.show()


def ircssmd(x,y,yer,lcb1,lcb2,lcb3,ucb1,ucb2,ucb3):
    ax = sns.set(style="white", color_codes=True)
    ax = plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
    ax = subplot(2,2,1)
    ax = ylim(0,2.5)
    x, y = pd.Series(c_ssmd[np.where(c_flag<0.9)], name="IRC0218 $\Sigma_{1kpc}$ Q"), pd.Series(c_dk[np.where(c_flag<0.9)], name="D4000")
    yer = c_dk_err[np.where(c_flag<0.9)]
    sort = np.argsort(x)
    x = x[sort]
    x = np.array(x)
    y = y[sort] 
    y = np.array(y)
    yer = yer[sort]
    yer = np.array(yer)
    xer=zeros(len(x))
    cov=zeros(len(x))   # no correlation between error measurements
    i = 0
    nboot=10000   # number of bootstrapping trials
    def func(x): return x[1]*x[0]+x[2]
    a,b,aerr,berr,covab=bces.bces.bcesp(x,xer,y,yer,cov,nboot)
    ybces=a[3]*x+b[3]  # the integer corresponds to the desired BCES method for plotting (3-ort, 0-y|x, 1-x|y, *don't use bissector*)
    # array with best-fit parameters
    fitm=np.array([ a[i],b[i] ])
    # covariance matrix of parameter uncertainties
    covm=np.array([ (aerr[i]**2,covab[i]), (covab[i],berr[i]**2) ])
    # Gets lower and upper bounds on the confidence band 
    lcb1,ucb1,x=nmmn.stats.confband(x, y, a[i], b[i], 0.68, x)
    lcb2,ucb2,x2=nmmn.stats.confband(x, y, a[i], b[i], 0.95, x)
    lcb3,ucb3,x3=nmmn.stats.confband(x, y, a[i], b[i], 0.997, x)
    errorbar(x,y,yerr=yer,fmt='o')
    ax = plot(x,ybces,'-k')
    ax = fill_between(x, lcb1, ucb1, alpha=0.6, facecolor='purple')
    ax = fill_between(x, lcb2, ucb2, alpha=0.3, facecolor='blue')
    ax = fill_between(x, lcb3, ucb3, alpha=0.4, facecolor='grey')
    ax = xlabel('IRC0218 $\Sigma_{1kpc}$ Q')
    ax = ylabel('D4000')
    
    ax = subplot(2,2,2)
    ax = ylim(0,2.5)
    x, y = pd.Series(c_ssmd[np.where(c_flag>0.9)], name="IRC0218 $\Sigma_{1kpc}$ SF"), pd.Series(c_dk[np.where(c_flag>0.9)], name="D4000")
    yer = c_dk_err[np.where(c_flag>0.9)]
    sort = np.argsort(x)
    x = x[sort]
    x = np.array(x)
    y = y[sort] 
    y = np.array(y)
    yer = yer[sort]
    yer = np.array(yer)
    xer=zeros(len(x))
    cov=zeros(len(x))   # no correlation between error measurements
    i = 0
    nboot=10000   # number of bootstrapping trials
    def func(x): return x[1]*x[0]+x[2]
    a,b,aerr,berr,covab=bces.bces.bcesp(x,xer,y,yer,cov,nboot)
    ybces=a[3]*x+b[3]  # the integer corresponds to the desired BCES method for plotting (3-ort, 0-y|x, 1-x|y, *don't use bissector*)
    # array with best-fit parameters
    fitm=np.array([ a[i],b[i] ])
    # covariance matrix of parameter uncertainties
    covm=np.array([ (aerr[i]**2,covab[i]), (covab[i],berr[i]**2) ])
    # Gets lower and upper bounds on the confidence band 
    lcb1,ucb1,x=nmmn.stats.confband(x, y, a[i], b[i], 0.68, x)
    lcb2,ucb2,x2=nmmn.stats.confband(x, y, a[i], b[i], 0.95, x)
    lcb3,ucb3,x3=nmmn.stats.confband(x, y, a[i], b[i], 0.997, x)
    errorbar(x,y,yerr=yer,fmt='o')
    ax = plot(x,ybces,'-k')
    ax = fill_between(x, lcb1, ucb1, alpha=0.6, facecolor='purple')
    ax = fill_between(x, lcb2, ucb2, alpha=0.3, facecolor='blue')
    ax = fill_between(x, lcb3, ucb3, alpha=0.4, facecolor='grey')
    ax = xlabel('IRC0218 $\Sigma_{1kpc}$ SF')
    ax = ylabel('D4000')
    purple_patch = mpatches.Patch(color='purple', label='68%')
    blue_patch = mpatches.Patch(color='blue', label='95%', alpha = 0.3)
    grey_patch = mpatches.Patch(color='grey', label='99.7%', alpha=0.4)
    plt.legend(handles=[purple_patch,blue_patch,grey_patch])


    
    ax = subplot(2,2,3)
    ax = ylim(0,2.5)
    x, y = pd.Series(c_ssmd[np.where(np.logical_and(c_flag<0.9,c_lmass>10.20))], name="IRC0218 $\Sigma_{1kpc}$ Q Mass Complete"), pd.Series(c_dk[np.where(np.logical_and(c_flag<0.9,c_lmass>10.20))], name="D4000")
    yer = c_dk_err[np.where(np.logical_and(c_flag<0.9,c_lmass>10.20))]
    sort = np.argsort(x)
    x = x[sort]
    x = np.array(x)
    y = y[sort] 
    y = np.array(y)
    yer = yer[sort]
    yer = np.array(yer)
    xer=zeros(len(x))
    cov=zeros(len(x))   # no correlation between error measurements
    i = 0
    nboot=10000   # number of bootstrapping trials
    def func(x): return x[1]*x[0]+x[2]
    a,b,aerr,berr,covab=bces.bces.bcesp(x,xer,y,yer,cov,nboot)
    ybces=a[3]*x+b[3]  # the integer corresponds to the desired BCES method for plotting (3-ort, 0-y|x, 1-x|y, *don't use bissector*)
    # array with best-fit parameters
    fitm=np.array([ a[i],b[i] ])
    # covariance matrix of parameter uncertainties
    covm=np.array([ (aerr[i]**2,covab[i]), (covab[i],berr[i]**2) ])
    # Gets lower and upper bounds on the confidence band 
    lcb1,ucb1,x=nmmn.stats.confband(x, y, a[i], b[i], 0.68, x)
    lcb2,ucb2,x2=nmmn.stats.confband(x, y, a[i], b[i], 0.95, x)
    lcb3,ucb3,x3=nmmn.stats.confband(x, y, a[i], b[i], 0.997, x)
    errorbar(x,y,yerr=yer,fmt='o')
    ax = plot(x,ybces,'-k')    
    ax = fill_between(x, lcb1, ucb1, alpha=0.6, facecolor='purple')
    ax = fill_between(x, lcb2, ucb2, alpha=0.3, facecolor='blue')
    ax = fill_between(x, lcb3, ucb3, alpha=0.4, facecolor='grey')
    ax = xlabel('IRC0218 $\Sigma_{1kpc}$ Q Mass Complete')
    ax = ylabel('D4000')
    
    ax = subplot(2,2,4)
    ax = ylim(0,2.5)
    x, y = pd.Series(c_ssmd[np.where(np.logical_and(c_flag>0.9,c_lmass>10.20))], name="IRC0218 $\Sigma_{1kpc}$ SF Mass Complete"), pd.Series(c_dk[np.where(np.logical_and(c_flag>0.9,c_lmass>10.20))], name="D4000")
    yer = c_dk_err[np.where(np.logical_and(c_flag>0.9,c_lmass>10.20))]
    sort = np.argsort(x)
    x = x[sort]
    x = np.array(x)
    y = y[sort] 
    y = np.array(y)
    yer = yer[sort]
    yer = np.array(yer)
    xer=zeros(len(x))
    cov=zeros(len(x))   # no correlation between error measurements
    i = 0
    nboot=10000   # number of bootstrapping trials
    def func(x): return x[1]*x[0]+x[2]
    a,b,aerr,berr,covab=bces.bces.bcesp(x,xer,y,yer,cov,nboot)
    ybces=a[3]*x+b[3]  # the integer corresponds to the desired BCES method for plotting (3-ort, 0-y|x, 1-x|y, *don't use bissector*)
    # array with best-fit parameters
    fitm=np.array([ a[i],b[i] ])
    # covariance matrix of parameter uncertainties
    covm=np.array([ (aerr[i]**2,covab[i]), (covab[i],berr[i]**2) ])
    # Gets lower and upper bounds on the confidence band 
    lcb1,ucb1,x=nmmn.stats.confband(x, y, a[i], b[i], 0.68, x)
    lcb2,ucb2,x2=nmmn.stats.confband(x, y, a[i], b[i], 0.95, x)
    lcb3,ucb3,x3=nmmn.stats.confband(x, y, a[i], b[i], 0.997, x)
    errorbar(x,y,yerr=yer,fmt='o')
    ax = plot(x,a[i]*x+b[i],'-k')
    ax = fill_between(x, lcb1, ucb1, alpha=0.6, facecolor='purple')
    ax = fill_between(x, lcb2, ucb2, alpha=0.3, facecolor='blue')
    ax = fill_between(x, lcb3, ucb3, alpha=0.4, facecolor='grey')
    ax = xlabel('IRC0218 $\Sigma_{1kpc}$ SF Mass Complete')
    ax = ylabel('D4000')
    ax = plt.show()
    

def ircssmd_all(x,y,yer,lcb1,lcb2,lcb3,ucb1,ucb2,ucb3,xcb1):
    ax = sns.set(style="white", color_codes=True)
    ax = plt.figure(figsize=(18, 16), dpi= 80, facecolor='none', edgecolor='k')
    ax = subplot(2,2,1)
    #ax = ylim(0,2.5)
    x, y = pd.Series(c_ssmd), pd.Series(c_dk)
    yer = c_dk_err
    sort = np.argsort(x)
    x = x[sort]
    x = np.array(x)
    y = y[sort] 
    y = np.array(y)
    yer = yer[sort]
    yer = np.array(yer)
    xer=zeros(len(x))
    cov=zeros(len(x))   # no correlation between error measurements
    i = 0
    nboot=10000   # number of bootstrapping trials
    def func(x): return x[1]*x[0]+x[2]
    a,b,aerr,berr,covab=bces.bces.bcesp(x,xer,y,yer,cov,nboot)
    ybces=a[3]*x+b[3]  # the integer corresponds to the desired BCES method for plotting (3-ort, 0-y|x, 1-x|y, *don't use bissector*)
    # array with best-fit parameters
    fitm=np.array([ a[i],b[i] ])
    # covariance matrix of parameter uncertainties
    covm=np.array([ (aerr[i]**2,covab[i]), (covab[i],berr[i]**2) ])
    # Gets lower and upper bounds on the confidence band 
    lcb1,ucb1,x=nmmn.stats.confband(x, y, a[i], b[i], 0.68, x)
    lcb2,ucb2,x2=nmmn.stats.confband(x, y, a[i], b[i], 0.95, x)
    lcb3,ucb3,x3=nmmn.stats.confband(x, y, a[i], b[i], 0.997, x)
    errorbar(x,y,yerr=yer,fmt='o')
    ax = plot(x,ybces,'-k')
    ax = fill_between(x, lcb1, ucb1, alpha=0.6, facecolor='purple')
    ax = fill_between(x, lcb2, ucb2, alpha=0.3, facecolor='blue')
    ax = fill_between(x, lcb3, ucb3, alpha=0.4, facecolor='grey')
    ax = xlabel('IRC0218 $\Sigma_{1kpc}$ Q & SF')
    ax = ylabel('D4000')
    
    
    ax = subplot(2,2,2)
    #ax = ylim(0,2.5)
    x, y = pd.Series(c_ssmd[np.where(c_lmass>10.2)]), pd.Series(c_dk[np.where(c_lmass>10.2)])
    yer = c_dk_err[np.where(c_lmass>10.2)]
    sort = np.argsort(x)
    x = x[sort]
    x = np.array(x)
    y = y[sort] 
    y = np.array(y)
    yer = yer[sort]
    yer = np.array(yer)
    xer=zeros(len(x))
    cov=zeros(len(x))   # no correlation between error measurements
    i = 0
    nboot=10000   # number of bootstrapping trials
    def func(x): return x[1]*x[0]+x[2]
    a,b,aerr,berr,covab=bces.bces.bcesp(x,xer,y,yer,cov,nboot)
    ybces=a[3]*x+b[3]  # the integer corresponds to the desired BCES method for plotting (3-ort, 0-y|x, 1-x|y, *don't use bissector*)
    # array with best-fit parameters
    fitm=np.array([ a[i],b[i] ])
    # covariance matrix of parameter uncertainties
    covm=np.array([ (aerr[i]**2,covab[i]), (covab[i],berr[i]**2) ])
    # Gets lower and upper bounds on the confidence band 
    lcb1,ucb1,x=nmmn.stats.confband(x, y, a[i], b[i], 0.68, x)
    lcb2,ucb2,x2=nmmn.stats.confband(x, y, a[i], b[i], 0.95, x)
    lcb3,ucb3,x3=nmmn.stats.confband(x, y, a[i], b[i], 0.997, x)
    errorbar(x,y,yerr=yer,fmt='o')
    ax = plot(x,ybces,'-k')
    ax = fill_between(x, lcb1, ucb1, alpha=0.6, facecolor='purple')
    ax = fill_between(x, lcb2, ucb2, alpha=0.3, facecolor='blue')
    ax = fill_between(x, lcb3, ucb3, alpha=0.4, facecolor='grey')
    ax = xlabel('IRC0218 $\Sigma_{1kpc}$ Q & SF Mass Complete')
    ax = ylabel('D4000')
    
    
    
    plt.show()

def ircn_all(x,y,yer,lcb1,lcb2,lcb3,ucb1,ucb2,ucb3,xcb1):
    ax = sns.set(style="white", color_codes=True)
    ax = plt.figure(figsize=(18, 16), dpi= 80, facecolor='none', edgecolor='k')
    ax = subplot(2,2,1)
    #ax = ylim(0,2.5)
    x, y = pd.Series(c_n), pd.Series(c_dk)
    yer = c_dk_err
    sort = np.argsort(x)
    x = x[sort]
    x = np.array(x)
    y = y[sort] 
    y = np.array(y)
    yer = yer[sort]
    yer = np.array(yer)
    xer=zeros(len(x))
    cov=zeros(len(x))   # no correlation between error measurements
    i = 0
    nboot=10000   # number of bootstrapping trials
    def func(x): return x[1]*x[0]+x[2]
    a,b,aerr,berr,covab=bces.bces.bcesp(x,xer,y,yer,cov,nboot)
    ybces=a[3]*x+b[3]  # the integer corresponds to the desired BCES method for plotting (3-ort, 0-y|x, 1-x|y, *don't use bissector*)
    # array with best-fit parameters
    fitm=np.array([ a[i],b[i] ])
    # covariance matrix of parameter uncertainties
    covm=np.array([ (aerr[i]**2,covab[i]), (covab[i],berr[i]**2) ])
    # Gets lower and upper bounds on the confidence band 
    lcb1,ucb1,x=nmmn.stats.confband(x, y, a[i], b[i], 0.68, x)
    lcb2,ucb2,x2=nmmn.stats.confband(x, y, a[i], b[i], 0.95, x)
    lcb3,ucb3,x3=nmmn.stats.confband(x, y, a[i], b[i], 0.997, x)
    errorbar(x,y,yerr=yer,fmt='o')
    ax = plot(x,ybces,'-k')
    ax = fill_between(x, lcb1, ucb1, alpha=0.6, facecolor='purple')
    ax = fill_between(x, lcb2, ucb2, alpha=0.3, facecolor='blue')
    ax = fill_between(x, lcb3, ucb3, alpha=0.4, facecolor='grey')
    ax = xlabel('IRC0218 Sersic Index Q & SF')
    ax = ylabel('D4000')
    
    
    ax = subplot(2,2,2)
    #ax = ylim(0,2.5)
    x, y = pd.Series(c_n[np.where(c_lmass>10.2)]), pd.Series(c_dk[np.where(c_lmass>10.2)])
    yer = c_dk_err[np.where(c_lmass>10.2)]
    sort = np.argsort(x)
    x = x[sort]
    x = np.array(x)
    y = y[sort] 
    y = np.array(y)
    yer = yer[sort]
    yer = np.array(yer)
    xer=zeros(len(x))
    cov=zeros(len(x))   # no correlation between error measurements
    i = 0
    nboot=10000   # number of bootstrapping trials
    def func(x): return x[1]*x[0]+x[2]
    a,b,aerr,berr,covab=bces.bces.bcesp(x,xer,y,yer,cov,nboot)
    ybces=a[3]*x+b[3]  # the integer corresponds to the desired BCES method for plotting (3-ort, 0-y|x, 1-x|y, *don't use bissector*)
    # array with best-fit parameters
    fitm=np.array([ a[i],b[i] ])
    # covariance matrix of parameter uncertainties
    covm=np.array([ (aerr[i]**2,covab[i]), (covab[i],berr[i]**2) ])
    # Gets lower and upper bounds on the confidence band 
    lcb1,ucb1,x=nmmn.stats.confband(x, y, a[i], b[i], 0.68, x)
    lcb2,ucb2,x2=nmmn.stats.confband(x, y, a[i], b[i], 0.95, x)
    lcb3,ucb3,x3=nmmn.stats.confband(x, y, a[i], b[i], 0.997, x)
    errorbar(x,y,yerr=yer,fmt='o')
    ax = plot(x,ybces,'-k')
    ax = fill_between(x, lcb1, ucb1, alpha=0.6, facecolor='purple')
    ax = fill_between(x, lcb2, ucb2, alpha=0.3, facecolor='blue')
    ax = fill_between(x, lcb3, ucb3, alpha=0.4, facecolor='grey')
    ax = xlabel('IRC0218 Sersic Index Q & SF Mass Complete')
    ax = ylabel('D4000') 
    plt.show()


def ircr_all(x,y,yer,lcb1,lcb2,lcb3,ucb1,ucb2,ucb3,xcb1):
    ax = sns.set(style="white", color_codes=True)
    ax = plt.figure(figsize=(18, 16), dpi= 80, facecolor='none', edgecolor='k')
    ax = subplot(2,2,1)
    #ax = ylim(0,2.5)
    x, y = pd.Series(c_re_kpc), pd.Series(c_dk)
    yer = c_dk_err
    sort = np.argsort(x)
    x = x[sort]
    x = np.array(x)
    y = y[sort] 
    y = np.array(y)
    yer = yer[sort]
    yer = np.array(yer)
    xer=zeros(len(x))
    cov=zeros(len(x))   # no correlation between error measurements
    i = 0
    nboot=10000   # number of bootstrapping trials
    def func(x): return x[1]*x[0]+x[2]
    a,b,aerr,berr,covab=bces.bces.bcesp(x,xer,y,yer,cov,nboot)
    ybces=a[3]*x+b[3]  # the integer corresponds to the desired BCES method for plotting (3-ort, 0-y|x, 1-x|y, *don't use bissector*)
    # array with best-fit parameters
    fitm=np.array([ a[i],b[i] ])
    # covariance matrix of parameter uncertainties
    covm=np.array([ (aerr[i]**2,covab[i]), (covab[i],berr[i]**2) ])
    # Gets lower and upper bounds on the confidence band 
    lcb1,ucb1,x=nmmn.stats.confband(x, y, a[i], b[i], 0.68, x)
    lcb2,ucb2,x2=nmmn.stats.confband(x, y, a[i], b[i], 0.95, x)
    lcb3,ucb3,x3=nmmn.stats.confband(x, y, a[i], b[i], 0.997, x)
    errorbar(x,y,yerr=yer,fmt='o')
    ax = plot(x,ybces,'-k')
    ax = fill_between(x, lcb1, ucb1, alpha=0.6, facecolor='purple')
    ax = fill_between(x, lcb2, ucb2, alpha=0.3, facecolor='blue')
    ax = fill_between(x, lcb3, ucb3, alpha=0.4, facecolor='grey')
    ax = xlabel('IRC0218 Effective Radius kpc Q & SF')
    ax = ylabel('D4000')
    
    
    ax = subplot(2,2,2)
    #ax = ylim(0,2.5)
    x, y = pd.Series(c_re_kpc[np.where(c_lmass>10.2)]), pd.Series(c_dk[np.where(c_lmass>10.2)])
    yer = c_dk_err[np.where(c_lmass>10.2)]
    sort = np.argsort(x)
    x = x[sort]
    x = np.array(x)
    y = y[sort] 
    y = np.array(y)
    yer = yer[sort]
    yer = np.array(yer)
    xer=zeros(len(x))
    cov=zeros(len(x))   # no correlation between error measurements
    i = 0
    nboot=10000   # number of bootstrapping trials
    def func(x): return x[1]*x[0]+x[2]
    a,b,aerr,berr,covab=bces.bces.bcesp(x,xer,y,yer,cov,nboot)
    ybces=a[3]*x+b[3]  # the integer corresponds to the desired BCES method for plotting (3-ort, 0-y|x, 1-x|y, *don't use bissector*)
    # array with best-fit parameters
    fitm=np.array([ a[i],b[i] ])
    # covariance matrix of parameter uncertainties
    covm=np.array([ (aerr[i]**2,covab[i]), (covab[i],berr[i]**2) ])
    # Gets lower and upper bounds on the confidence band 
    lcb1,ucb1,x=nmmn.stats.confband(x, y, a[i], b[i], 0.68, x)
    lcb2,ucb2,x2=nmmn.stats.confband(x, y, a[i], b[i], 0.95, x)
    lcb3,ucb3,x3=nmmn.stats.confband(x, y, a[i], b[i], 0.997, x)
    errorbar(x,y,yerr=yer,fmt='o')
    ax = plot(x,ybces,'-k')
    ax = fill_between(x, lcb1, ucb1, alpha=0.6, facecolor='purple')
    ax = fill_between(x, lcb2, ucb2, alpha=0.3, facecolor='blue')
    ax = fill_between(x, lcb3, ucb3, alpha=0.4, facecolor='grey')
    ax = xlabel('IRC0218 Effective Radius kpc Q & SF Mass Complete')
    ax = ylabel('D4000')
    plt.show()

def ircm_all(x,y,yer,lcb1,lcb2,lcb3,ucb1,ucb2,ucb3,xcb1):
    ax = sns.set(style="white", color_codes=True)
    ax = plt.figure(figsize=(18, 16), dpi= 80, facecolor='none', edgecolor='k')
    ax = subplot(2,2,1)
    #ax = ylim(0,2.5)
    x, y = pd.Series(c_lmass), pd.Series(c_dk)
    yer = c_dk_err
    sort = np.argsort(x)
    x = x[sort]
    x = np.array(x)
    y = y[sort] 
    y = np.array(y)
    yer = yer[sort]
    yer = np.array(yer)
    xer=zeros(len(x))
    cov=zeros(len(x))   # no correlation between error measurements
    i = 0
    nboot=10000   # number of bootstrapping trials
    def func(x): return x[1]*x[0]+x[2]
    a,b,aerr,berr,covab=bces.bces.bcesp(x,xer,y,yer,cov,nboot)
    ybces=a[3]*x+b[3]  # the integer corresponds to the desired BCES method for plotting (3-ort, 0-y|x, 1-x|y, *don't use bissector*)
    # array with best-fit parameters
    fitm=np.array([ a[i],b[i] ])
    # covariance matrix of parameter uncertainties
    covm=np.array([ (aerr[i]**2,covab[i]), (covab[i],berr[i]**2) ])
    # Gets lower and upper bounds on the confidence band 
    lcb1,ucb1,x=nmmn.stats.confband(x, y, a[i], b[i], 0.68, x)
    lcb2,ucb2,x2=nmmn.stats.confband(x, y, a[i], b[i], 0.95, x)
    lcb3,ucb3,x3=nmmn.stats.confband(x, y, a[i], b[i], 0.997, x)
    errorbar(x,y,yerr=yer,fmt='o')
    ax = plot(x,ybces,'-k')
    ax = fill_between(x, lcb1, ucb1, alpha=0.6, facecolor='purple')
    ax = fill_between(x, lcb2, ucb2, alpha=0.3, facecolor='blue')
    ax = fill_between(x, lcb3, ucb3, alpha=0.4, facecolor='grey')
    ax = xlabel('IRC0218 LogMass Q & SF')
    ax = ylabel('D4000')
    
    
    ax = subplot(2,2,2)
    #ax = ylim(0,2.5)
    x, y = pd.Series(c_lmass[np.where(c_lmass>10.2)]), pd.Series(c_dk[np.where(c_lmass>10.2)])
    yer = c_dk_err[np.where(c_lmass>10.2)]
    sort = np.argsort(x)
    x = x[sort]
    x = np.array(x)
    y = y[sort] 
    y = np.array(y)
    yer = yer[sort]
    yer = np.array(yer)
    xer=zeros(len(x))
    cov=zeros(len(x))   # no correlation between error measurements
    i = 0
    nboot=10000   # number of bootstrapping trials
    def func(x): return x[1]*x[0]+x[2]
    a,b,aerr,berr,covab=bces.bces.bcesp(x,xer,y,yer,cov,nboot)
    ybces=a[3]*x+b[3]  # the integer corresponds to the desired BCES method for plotting (3-ort, 0-y|x, 1-x|y, *don't use bissector*)
    # array with best-fit parameters
    fitm=np.array([ a[i],b[i] ])
    # covariance matrix of parameter uncertainties
    covm=np.array([ (aerr[i]**2,covab[i]), (covab[i],berr[i]**2) ])
    # Gets lower and upper bounds on the confidence band 
    lcb1,ucb1,x=nmmn.stats.confband(x, y, a[i], b[i], 0.68, x)
    lcb2,ucb2,x2=nmmn.stats.confband(x, y, a[i], b[i], 0.95, x)
    lcb3,ucb3,x3=nmmn.stats.confband(x, y, a[i], b[i], 0.997, x)
    errorbar(x,y,yerr=yer,fmt='o')
    ax = plot(x,ybces,'-k')
    ax = fill_between(x, lcb1, ucb1, alpha=0.6, facecolor='purple')
    ax = fill_between(x, lcb2, ucb2, alpha=0.3, facecolor='blue')
    ax = fill_between(x, lcb3, ucb3, alpha=0.4, facecolor='grey')
    ax = xlabel('IRC0218 LogMass Q & SF Mass Complete')
    ax = ylabel('D4000')
    plt.show()

    
def fieldm(x,y,yer,lcb1,lcb2,lcb3,ucb1,ucb2,ucb3):
    ax=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
    ax = sns.set(style="white", color_codes=True)
    ax = subplot(2,2,1)
    ax = ylim(0,5)
    x, y = pd.Series(f_lmass[np.where(f_flag>0.9)]), pd.Series(f_dk[np.where(f_flag>0.9)])
    yer = f_dk_err[np.where(f_flag>0.9)]
    sort = np.argsort(x)
    x = x[sort]
    x = np.array(x)
    y = y[sort] 
    y = np.array(y)
    yer = yer[sort]
    yer = np.array(yer)
    xer=zeros(len(x))
    cov=zeros(len(x))   # no correlation between error measurements
    i = 0
    nboot=10000   # number of bootstrapping trials
    def func(x): return x[1]*x[0]+x[2]
    a,b,aerr,berr,covab=bces.bces.bcesp(x,xer,y,yer,cov,nboot)
    ybces=a[3]*x+b[3]  # the integer corresponds to the desired BCES method for plotting (3-ort, 0-y|x, 1-x|y, *don't use bissector*)
    # array with best-fit parameters
    fitm=np.array([ a[i],b[i] ])
    # covariance matrix of parameter uncertainties
    covm=np.array([ (aerr[i]**2,covab[i]), (covab[i],berr[i]**2) ])
    # Gets lower and upper bounds on the confidence band 
    lcb1,ucb1,x=nmmn.stats.confband(x, y, a[i], b[i], 0.68, x)
    lcb2,ucb2,x2=nmmn.stats.confband(x, y, a[i], b[i], 0.95, x)
    lcb3,ucb3,x3=nmmn.stats.confband(x, y, a[i], b[i], 0.997, x)
    errorbar(x,y,yerr=yer,fmt='o')
    ax = plot(x,ybces,'-k')
    ax = fill_between(x, lcb1, ucb1, alpha=0.6, facecolor='purple')
    ax = fill_between(x, lcb2, ucb2, alpha=0.3, facecolor='blue')
    ax = fill_between(x, lcb3, ucb3, alpha=0.4, facecolor='grey')
    #ax = bar(x,y,yerr=[f_dk_p[np.where(f_flag>0.9)], f_dk_n[np.where(f_flag>0.9)]], facecolor='none')
    ax = legend(loc='best')
    ax = xlabel('Field LogMass SF')
    ax = ylabel('$D4000$')
    
    ax = subplot(2,2,2)
   # ax = xlim(0.4,5)
   # ax = ylim(-1,5)
    x, y = pd.Series(f_lmass[np.where(f_flag<0.9)]), pd.Series(f_dk[np.where(f_flag<0.9)])
    yer = pd.Series(f_dk_err[np.where(f_flag<0.9)])
    sort = np.argsort(x)
    x = x[sort]
    x = np.array(x)
    y = y[sort] 
    y = np.array(y)
    yer = yer[sort]
    yer = np.array(yer)
    xer=zeros(len(x))
    cov=zeros(len(x))   # no correlation between error measurements
    i = 0
    nboot=10000   # number of bootstrapping trials
    def func(x): return x[1]*x[0]+x[2]
    a,b,aerr,berr,covab=bces.bces.bcesp(x,xer,y,yer,cov,nboot)
    ybces=a[3]*x+b[3]  # the integer corresponds to the desired BCES method for plotting (3-ort, 0-y|x, 1-x|y, *don't use bissector*)
    # array with best-fit parameters
    fitm=np.array([ a[i],b[i] ])
    # covariance matrix of parameter uncertainties
    covm=np.array([ (aerr[i]**2,covab[i]), (covab[i],berr[i]**2) ])
    # Gets lower and upper bounds on the confidence band 
    lcb1,ucb1,x=nmmn.stats.confband(x, y, a[i], b[i], 0.68, x)
    lcb2,ucb2,x2=nmmn.stats.confband(x, y, a[i], b[i], 0.95, x)
    lcb3,ucb3,x3=nmmn.stats.confband(x, y, a[i], b[i], 0.997, x)
    errorbar(x,y,yerr=yer,fmt='o')
    ax = plot(x,ybces,'-k')
    ax = fill_between(x, lcb1, ucb1, alpha=0.6, facecolor='purple')
    ax = fill_between(x, lcb2, ucb2, alpha=0.3, facecolor='blue')
    ax = fill_between(x, lcb3, ucb3, alpha=0.4, facecolor='grey')
    ax = xlabel('Field LogMass Q')
    purple_patch = mpatches.Patch(color='purple', label='68%')
    blue_patch = mpatches.Patch(color='blue', label='95%', alpha = 0.3)
    grey_patch = mpatches.Patch(color='grey', label='99.7%', alpha=0.4)
    plt.legend(handles=[purple_patch,blue_patch,grey_patch])
    
    
    
    ax = subplot(2,2,3)
   ## ax = xlim(0.4,4)
    #ax = ylim(-1,5)
    x, y = pd.Series(f_lmass[np.where(np.logical_and(f_flag>0.9,f_lmass>10.20))]), pd.Series(f_dk[np.where(np.logical_and(f_flag>0.9,f_lmass>10.20))])
    yer = pd.Series(f_dk_err[np.where(np.logical_and(f_flag>0.9,f_lmass>10.20))])
    sort = np.argsort(x)
    x = x[sort]
    x = np.array(x)
    y = y[sort] 
    y = np.array(y)
    yer = yer[sort]
    yer = np.array(yer)
    xer=zeros(len(x))
    cov=zeros(len(x))   # no correlation between error measurements
    i = 0
    nboot=10000   # number of bootstrapping trials
    def func(x): return x[1]*x[0]+x[2]
    a,b,aerr,berr,covab=bces.bces.bcesp(x,xer,y,yer,cov,nboot)
    ybces=a[3]*x+b[3]  # the integer corresponds to the desired BCES method for plotting (3-ort, 0-y|x, 1-x|y, *don't use bissector*)
    # array with best-fit parameters
    fitm=np.array([ a[i],b[i] ])
    # covariance matrix of parameter uncertainties
    covm=np.array([ (aerr[i]**2,covab[i]), (covab[i],berr[i]**2) ])
    # Gets lower and upper bounds on the confidence band 
    lcb1,ucb1,x=nmmn.stats.confband(x, y, a[i], b[i], 0.68, x)
    lcb2,ucb2,x2=nmmn.stats.confband(x, y, a[i], b[i], 0.95, x)
    lcb3,ucb3,x3=nmmn.stats.confband(x, y, a[i], b[i], 0.997, x)
    errorbar(x,y,yerr=yer,fmt='o')
    ax = plot(x,ybces,'-k')
    ax = fill_between(x, lcb1, ucb1, alpha=0.6, facecolor='purple')
    ax = fill_between(x, lcb2, ucb2, alpha=0.3, facecolor='blue')
    ax = fill_between(x, lcb3, ucb3, alpha=0.4, facecolor='grey')
    ax = xlabel('Field LogMass SF Mass Complete')
    ax = ylabel('D4000')
    ax = plt.show()

def ircm(x,y,yer,lcb1,lcb2,lcb3,ucb1,ucb2,ucb3):
    ax = sns.set(style="white", color_codes=True)
    ax = plt.figure(figsize=(18, 16), dpi= 80, facecolor='none', edgecolor='k')
    ax = subplot(2,2,1)
    #ax = ylim(0,2.5)
    x = c_lmass[np.where(c_flag>0.9)]
    y = c_dk[np.where(c_flag>0.9)]
    yer = c_dk_err[np.where(c_flag>0.9)]
    sort = np.argsort(x)
    x = x[sort]
    x = np.array(x)
    y = y[sort] 
    y = np.array(y)
    yer = yer[sort]
    yer = np.array(yer)
    xer=zeros(len(x))
    cov=zeros(len(x))   # no correlation between error measurements
    i = 0
    nboot=10000   # number of bootstrapping trials
    def func(x): return x[1]*x[0]+x[2]
    a,b,aerr,berr,covab=bces.bces.bcesp(x,xer,y,yer,cov,nboot)
    ybces=a[3]*x+b[3]  # the integer corresponds to the desired BCES method for plotting (3-ort, 0-y|x, 1-x|y, *don't use bissector*)
    # array with best-fit parameters
    fitm=np.array([ a[i],b[i] ])
    # covariance matrix of parameter uncertainties
    covm=np.array([ (aerr[i]**2,covab[i]), (covab[i],berr[i]**2) ])
    # Gets lower and upper bounds on the confidence band 
    lcb1,ucb1,x=nmmn.stats.confband(x, y, a[i], b[i], 0.68, x)
    lcb2,ucb2,x2=nmmn.stats.confband(x, y, a[i], b[i], 0.95, x)
    lcb3,ucb3,x3=nmmn.stats.confband(x, y, a[i], b[i], 0.997, x)
    errorbar(x,y,yerr=yer,fmt='o')
    ax = plot(x,ybces,'-k')
    ax = fill_between(x, lcb1, ucb1, alpha=0.6, facecolor='purple')
    ax = fill_between(x, lcb2, ucb2, alpha=0.3, facecolor='blue')
    ax = fill_between(x, lcb3, ucb3, alpha=0.4, facecolor='grey')
    ax = xlabel('IRC0218 LogMass SF')
    ax = ylabel('D4000')
    
    
    ax = subplot(2,2,2)
    #ax = ylim(0,2.5)
    x = c_lmass[np.where(c_flag<0.9)]
    y = c_dk[np.where(c_flag<0.9)]
    yer = c_dk_err[np.where(c_flag<0.9)]
    sort = np.argsort(x)
    x = x[sort]
    x = np.array(x)
    y = y[sort] 
    y = np.array(y)
    yer = yer[sort]
    yer = np.array(yer)
    xer=zeros(len(x))
    cov=zeros(len(x))   # no correlation between error measurements
    i = 0
    nboot=10000   # number of bootstrapping trials
    def func(x): return x[1]*x[0]+x[2]
    a,b,aerr,berr,covab=bces.bces.bcesp(x,xer,y,yer,cov,nboot)
    ybces=a[3]*x+b[3]  # the integer corresponds to the desired BCES method for plotting (3-ort, 0-y|x, 1-x|y, *don't use bissector*)
    # array with best-fit parameters
    fitm=np.array([ a[i],b[i] ])
    # covariance matrix of parameter uncertainties
    covm=np.array([ (aerr[i]**2,covab[i]), (covab[i],berr[i]**2) ])
    # Gets lower and upper bounds on the confidence band 
    lcb1,ucb1,x=nmmn.stats.confband(x, y, a[i], b[i], 0.68, x)
    lcb2,ucb2,x2=nmmn.stats.confband(x, y, a[i], b[i], 0.95, x)
    lcb3,ucb3,x3=nmmn.stats.confband(x, y, a[i], b[i], 0.997, x)
    errorbar(x,y,yerr=yer,fmt='o')
    ax = plot(x,ybces,'-k')
    ax = fill_between(x, lcb1, ucb1, alpha=0.6, facecolor='purple')
    ax = fill_between(x, lcb2, ucb2, alpha=0.3, facecolor='blue')
    ax = fill_between(x, lcb3, ucb3, alpha=0.4, facecolor='grey')
    ax = xlabel('IRC0218 LogMass Q')
    ax = ylabel('D4000')
    
    ax = subplot(2,2,3)
    x = np.array([ 10.22,  10.25,  10.31,  10.76,  10.83,  10.89,  10.89])
    y = np.array([ 1.48,  1.14,  1.22,  1.43,  1.16,  1.43,  1.15])
    yer = np.array([ 1.395,  0.1  ,  0.155,  0.235,  0.275,  0.135,  0.12 ])
    xer=zeros(len(x))
    cov=zeros(len(x))   # no correlation between error measurements
    i=0 
    nboot = 10000
    a,b,erra,errb,covab=bces.bces.bcesp(x,xer,y,yer,cov,nboot)
    a = np.array([ -2.01088498e-02,  -3.34308960e+01,   1.28590319e-01,3.38432765e-02])
    b = np.array([  1.50112872e+00,   3.56210472e+02,  -7.18932240e-02,9.29291802e-01])
    ybces=a[3]*x+b[3]  # the integer corresponds to the desired BCES method for plotting (3-ort, 0-y|x, 1-x|y, *don't use bissector*)
        # array with best-fit parameters
    fitm=np.array([ a[i],b[i] ])
        # covariance matrix of parameter uncertainties
    covm=np.array([ (aerr[i]**2,covab[i]), (covab[i],berr[i]**2) ])
        # Gets lower and upper bounds on the confidence band 
    lcb1,ucb1,x=nmmn.stats.confband(x, y, a[i], b[i], 0.68, x)
    lcb2,ucb2,x2=nmmn.stats.confband(x, y, a[i], b[i], 0.95, x)
    lcb3,ucb3,x3=nmmn.stats.confband(x, y, a[i], b[i], 0.997, x)
    errorbar(x,y,yerr=yer,fmt='o')
    ax = plot(x,ybces,'-k')
    ax = fill_between(x, lcb1, ucb1, alpha=0.6, facecolor='purple')
    ax = fill_between(x, lcb2, ucb2, alpha=0.3, facecolor='blue')
    ax = fill_between(x, lcb3, ucb3, alpha=0.4, facecolor='grey')
    ax = xlabel('IRC0218 LogMass SF Mass Complete')
    ax = ylabel('D4000')
    
    
    ax = subplot(2,2,4)
    #ax = ylim(0,2.5)
    x = c_lmass[np.where(np.logical_and(c_flag<0.9,c_lmass>10.20))]
    y = c_dk[np.where(np.logical_and(c_flag<0.9,c_lmass>10.20))]
    yer = c_dk_err[np.where(np.logical_and(c_flag<0.9,c_lmass>10.20))]
    sort = np.argsort(x)
    x = x[sort]
    x = np.array(x)
    y = y[sort] 
    y = np.array(y)
    yer = yer[sort]
    yer = np.array(yer)
    xer=zeros(len(x))
    cov=zeros(len(x))   # no correlation between error measurements
    i = 0
    nboot=10000   # number of bootstrapping trials
    def func(x): return x[1]*x[0]+x[2]
    a,b,aerr,berr,covab=bces.bces.bcesp(x,xer,y,yer,cov,nboot)
    ybces=a[3]*x+b[3]  # the integer corresponds to the desired BCES method for plotting (3-ort, 0-y|x, 1-x|y, *don't use bissector*)
    # array with best-fit parameters
    fitm=np.array([ a[i],b[i] ])
    # covariance matrix of parameter uncertainties
    covm=np.array([ (aerr[i]**2,covab[i]), (covab[i],berr[i]**2) ])
    # Gets lower and upper bounds on the confidence band 
    lcb1,ucb1,x=nmmn.stats.confband(x, y, a[i], b[i], 0.68, x)
    lcb2,ucb2,x2=nmmn.stats.confband(x, y, a[i], b[i], 0.95, x)
    lcb3,ucb3,x3=nmmn.stats.confband(x, y, a[i], b[i], 0.997, x)
    errorbar(x,y,yerr=yer,fmt='o')
    ax = plot(x,ybces,'-k')
    ax = fill_between(x, lcb1, ucb1, alpha=0.6, facecolor='purple')
    ax = fill_between(x, lcb2, ucb2, alpha=0.3, facecolor='blue')
    ax = fill_between(x, lcb3, ucb3, alpha=0.4, facecolor='grey')
    ax = xlabel('IRC0218 LogMass Q Mass Complete')
    ax = ylabel('D4000')    
    
    plt.show()

def ircmr(x,y,yer,lcb1,lcb2,lcb3,ucb1,ucb2,ucb3):
    ax = sns.set(style="white", color_codes=True)
    ax = plt.figure(figsize=(18, 16), dpi= 80, facecolor='none', edgecolor='k')
    ax = subplot(2,2,1)
    #ax = ylim(0,2.5)
    x = c_lmass[np.where(c_flag>0.9)]/c_re_kpc[np.where(c_flag>0.9)]
    y = c_dk[np.where(c_flag>0.9)]
    yer = c_dk_err[np.where(c_flag>0.9)]
    sort = np.argsort(x)
    x = x[sort]
    x = np.array(x)
    y = y[sort] 
    y = np.array(y)
    yer = yer[sort]
    yer = np.array(yer)
    xer=zeros(len(x))
    cov=zeros(len(x))   # no correlation between error measurements
    i = 0
    nboot=10000   # number of bootstrapping trials
    def func(x): return x[1]*x[0]+x[2]
    a,b,aerr,berr,covab=bces.bces.bcesp(x,xer,y,yer,cov,nboot)
    ybces=a[3]*x+b[3]  # the integer corresponds to the desired BCES method for plotting (3-ort, 0-y|x, 1-x|y, *don't use bissector*)
    # array with best-fit parameters
    fitm=np.array([ a[i],b[i] ])
    # covariance matrix of parameter uncertainties
    covm=np.array([ (aerr[i]**2,covab[i]), (covab[i],berr[i]**2) ])
    # Gets lower and upper bounds on the confidence band 
    lcb1,ucb1,x=nmmn.stats.confband(x, y, a[i], b[i], 0.68, x)
    lcb2,ucb2,x2=nmmn.stats.confband(x, y, a[i], b[i], 0.95, x)
    lcb3,ucb3,x3=nmmn.stats.confband(x, y, a[i], b[i], 0.997, x)
    errorbar(x,y,yerr=yer,fmt='o')
    ax = plot(x,ybces,'-k')
    ax = fill_between(x, lcb1, ucb1, alpha=0.6, facecolor='purple')
    ax = fill_between(x, lcb2, ucb2, alpha=0.3, facecolor='blue')
    ax = fill_between(x, lcb3, ucb3, alpha=0.4, facecolor='grey')
    ax = xlabel('IRC0218 LogMass/R SF')
    ax = ylabel('D4000')
    
    
    ax = subplot(2,2,2)
    #ax = ylim(0,2.5)
    x = c_lmass[np.where(c_flag<0.9)]/c_re_kpc[np.where(c_flag<0.9)]
    y = c_dk[np.where(c_flag<0.9)]
    yer = c_dk_err[np.where(c_flag<0.9)]
    sort = np.argsort(x)
    x = x[sort]
    x = np.array(x)
    y = y[sort] 
    y = np.array(y)
    yer = yer[sort]
    yer = np.array(yer)
    xer=zeros(len(x))
    cov=zeros(len(x))   # no correlation between error measurements
    i = 0
    nboot=10000   # number of bootstrapping trials
    def func(x): return x[1]*x[0]+x[2]
    a,b,aerr,berr,covab=bces.bces.bcesp(x,xer,y,yer,cov,nboot)
    ybces=a[3]*x+b[3]  # the integer corresponds to the desired BCES method for plotting (3-ort, 0-y|x, 1-x|y, *don't use bissector*)
    # array with best-fit parameters
    fitm=np.array([ a[i],b[i] ])
    # covariance matrix of parameter uncertainties
    covm=np.array([ (aerr[i]**2,covab[i]), (covab[i],berr[i]**2) ])
    # Gets lower and upper bounds on the confidence band 
    lcb1,ucb1,x=nmmn.stats.confband(x, y, a[i], b[i], 0.68, x)
    lcb2,ucb2,x2=nmmn.stats.confband(x, y, a[i], b[i], 0.95, x)
    lcb3,ucb3,x3=nmmn.stats.confband(x, y, a[i], b[i], 0.997, x)
    errorbar(x,y,yerr=yer,fmt='o')
    ax = plot(x,ybces,'-k')
    ax = fill_between(x, lcb1, ucb1, alpha=0.6, facecolor='purple')
    ax = fill_between(x, lcb2, ucb2, alpha=0.3, facecolor='blue')
    ax = fill_between(x, lcb3, ucb3, alpha=0.4, facecolor='grey')
    ax = xlabel('IRC0218 LogMass/R Q')
    ax = ylabel('D4000')
    
    ax = subplot(2,2,3)
    #x = np.array([  0.0630552 ,   0.13834198,   0.15097889,   1.50982026,    2.83197654,   6.79663764,  10.44563709])
    #y = np.array([ 1.48,  1.14,  1.16,  1.15,  1.43,  1.43,  1.22])
    #yer = np.array([ 1.395,  0.1  ,  0.275,  0.12 ,  0.235,  0.135,  0.155])
    x = c_lmass[np.where(np.logical_and(c_flag>0.9,c_lmass>10.20))]/c_re_kpc[np.where(np.logical_and(c_flag>0.9,c_lmass>10.20))]
    y = c_dk[np.where(np.logical_and(c_flag>0.9,c_lmass>10.20))]
    yer = c_dk_err[np.where(np.logical_and(c_flag>0.9,c_lmass>10.20))]
    sort = np.argsort(x)
    x = x[sort]
    x = np.array(x)
    y = y[sort] 
    y = np.array(y)
    yer = yer[sort]
    yer = np.array(yer)
    xer=zeros(len(x))
    cov=zeros(len(x))   # no correlation between error measurements
    i=0 
    nboot = 10000
    a,b,erra,errb,covab=bces.bces.bcesp(x,xer,y,yer,cov,nboot)
    #a = np.array([ -2.01088498e-02,  -3.34308960e+01,   1.28590319e-01,3.38432765e-02])
    #b = np.array([  1.50112872e+00,   3.56210472e+02,  -7.18932240e-02,9.29291802e-01])
    ybces=a[3]*x+b[3]  # the integer corresponds to the desired BCES method for plotting (3-ort, 0-y|x, 1-x|y, *don't use bissector*)
        # array with best-fit parameters
    fitm=np.array([ a[i],b[i] ])
        # covariance matrix of parameter uncertainties
    covm=np.array([ (aerr[i]**2,covab[i]), (covab[i],berr[i]**2) ])
        # Gets lower and upper bounds on the confidence band 
    lcb1,ucb1,x=nmmn.stats.confband(x, y, a[i], b[i], 0.68, x)
    lcb2,ucb2,x2=nmmn.stats.confband(x, y, a[i], b[i], 0.95, x)
    lcb3,ucb3,x3=nmmn.stats.confband(x, y, a[i], b[i], 0.997, x)
    errorbar(x,y,yerr=yer,fmt='o')
    ax = plot(x,ybces,'-k')
    ax = fill_between(x, lcb1, ucb1, alpha=0.6, facecolor='purple')
    ax = fill_between(x, lcb2, ucb2, alpha=0.3, facecolor='blue')
    ax = fill_between(x, lcb3, ucb3, alpha=0.4, facecolor='grey')
    ax = xlabel('IRC0218 LogMass/R SF Mass Complete')
    ax = ylabel('D4000')
    
    
    ax = subplot(2,2,4)
    #ax = ylim(0,2.5)
    x = c_lmass[np.where(np.logical_and(c_flag<0.9,c_lmass>10.20))]/c_re_kpc[np.where(np.logical_and(c_flag<0.9,c_lmass>10.20))]
    y = c_dk[np.where(np.logical_and(c_flag<0.9,c_lmass>10.20))]
    yer = c_dk_err[np.where(np.logical_and(c_flag<0.9,c_lmass>10.20))]
    sort = np.argsort(x)
    x = x[sort]
    x = np.array(x)
    y = y[sort] 
    y = np.array(y)
    yer = yer[sort]
    yer = np.array(yer)
    xer=zeros(len(x))
    cov=zeros(len(x))   # no correlation between error measurements
    i = 0
    nboot=10000   # number of bootstrapping trials
    def func(x): return x[1]*x[0]+x[2]
    a,b,aerr,berr,covab=bces.bces.bcesp(x,xer,y,yer,cov,nboot)
    ybces=a[3]*x+b[3]  # the integer corresponds to the desired BCES method for plotting (3-ort, 0-y|x, 1-x|y, *don't use bissector*)
    # array with best-fit parameters
    fitm=np.array([ a[i],b[i] ])
    # covariance matrix of parameter uncertainties
    covm=np.array([ (aerr[i]**2,covab[i]), (covab[i],berr[i]**2) ])
    # Gets lower and upper bounds on the confidence band 
    lcb1,ucb1,x=nmmn.stats.confband(x, y, a[i], b[i], 0.68, x)
    lcb2,ucb2,x2=nmmn.stats.confband(x, y, a[i], b[i], 0.95, x)
    lcb3,ucb3,x3=nmmn.stats.confband(x, y, a[i], b[i], 0.997, x)
    errorbar(x,y,yerr=yer,fmt='o')
    ax = plot(x,ybces,'-k')
    ax = fill_between(x, lcb1, ucb1, alpha=0.6, facecolor='purple')
    ax = fill_between(x, lcb2, ucb2, alpha=0.3, facecolor='blue')
    ax = fill_between(x, lcb3, ucb3, alpha=0.4, facecolor='grey')
    ax = xlabel('IRC0218 LogMass/R Q Mass Complete')
    ax = ylabel('D4000')    
    
    plt.show()

def ircmr2(x,y,yer,lcb1,lcb2,lcb3,ucb1,ucb2,ucb3):
    ax = sns.set(style="white", color_codes=True)
    ax = plt.figure(figsize=(18, 16), dpi= 80, facecolor='none', edgecolor='k')
    ax = subplot(2,2,1)
    #ax = ylim(0,2.5)
    x = c_lmass[np.where(c_flag>0.9)]/c_re_kpc[np.where(c_flag>0.9)]**2
    y = c_dk[np.where(c_flag>0.9)]
    yer = c_dk_err[np.where(c_flag>0.9)]
    sort = np.argsort(x)
    x = x[sort]
    x = np.array(x)
    y = y[sort] 
    y = np.array(y)
    yer = yer[sort]
    yer = np.array(yer)
    xer=zeros(len(x))
    cov=zeros(len(x))   # no correlation between error measurements
    i = 0
    nboot=10000   # number of bootstrapping trials
    def func(x): return x[1]*x[0]+x[2]
    a,b,aerr,berr,covab=bces.bces.bcesp(x,xer,y,yer,cov,nboot)
    ybces=a[3]*x+b[3]  # the integer corresponds to the desired BCES method for plotting (3-ort, 0-y|x, 1-x|y, *don't use bissector*)
    # array with best-fit parameters
    fitm=np.array([ a[i],b[i] ])
    # covariance matrix of parameter uncertainties
    covm=np.array([ (aerr[i]**2,covab[i]), (covab[i],berr[i]**2) ])
    # Gets lower and upper bounds on the confidence band 
    lcb1,ucb1,x=nmmn.stats.confband(x, y, a[i], b[i], 0.68, x)
    lcb2,ucb2,x2=nmmn.stats.confband(x, y, a[i], b[i], 0.95, x)
    lcb3,ucb3,x3=nmmn.stats.confband(x, y, a[i], b[i], 0.997, x)
    errorbar(x,y,yerr=yer,fmt='o')
    ax = plot(x,ybces,'-k')
    ax = fill_between(x, lcb1, ucb1, alpha=0.6, facecolor='purple')
    ax = fill_between(x, lcb2, ucb2, alpha=0.3, facecolor='blue')
    ax = fill_between(x, lcb3, ucb3, alpha=0.4, facecolor='grey')
    ax = xlabel('IRC0218 LogMass/R$^2$ SF')
    ax = ylabel('D4000')
    
    
    ax = subplot(2,2,2)
    #ax = ylim(0,2.5)
    x = c_lmass[np.where(c_flag<0.9)]/c_re_kpc[np.where(c_flag<0.9)]**2
    y = c_dk[np.where(c_flag<0.9)]
    yer = c_dk_err[np.where(c_flag<0.9)]
    sort = np.argsort(x)
    x = x[sort]
    x = np.array(x)
    y = y[sort] 
    y = np.array(y)
    yer = yer[sort]
    yer = np.array(yer)
    xer=zeros(len(x))
    cov=zeros(len(x))   # no correlation between error measurements
    i = 0
    nboot=10000   # number of bootstrapping trials
    def func(x): return x[1]*x[0]+x[2]
    a,b,aerr,berr,covab=bces.bces.bcesp(x,xer,y,yer,cov,nboot)
    ybces=a[3]*x+b[3]  # the integer corresponds to the desired BCES method for plotting (3-ort, 0-y|x, 1-x|y, *don't use bissector*)
    # array with best-fit parameters
    fitm=np.array([ a[i],b[i] ])
    # covariance matrix of parameter uncertainties
    covm=np.array([ (aerr[i]**2,covab[i]), (covab[i],berr[i]**2) ])
    # Gets lower and upper bounds on the confidence band 
    lcb1,ucb1,x=nmmn.stats.confband(x, y, a[i], b[i], 0.68, x)
    lcb2,ucb2,x2=nmmn.stats.confband(x, y, a[i], b[i], 0.95, x)
    lcb3,ucb3,x3=nmmn.stats.confband(x, y, a[i], b[i], 0.997, x)
    errorbar(x,y,yerr=yer,fmt='o')
    ax = plot(x,ybces,'-k')
    ax = fill_between(x, lcb1, ucb1, alpha=0.6, facecolor='purple')
    ax = fill_between(x, lcb2, ucb2, alpha=0.3, facecolor='blue')
    ax = fill_between(x, lcb3, ucb3, alpha=0.4, facecolor='grey')
    ax = xlabel('IRC0218 LogMass/R$^2$ Q')
    ax = ylabel('D4000')
    
    ax = subplot(2,2,3)
    #x = np.array([  0.0630552 ,   0.13834198,   0.15097889,   1.50982026,    2.83197654,   6.79663764,  10.44563709])
    #y = np.array([ 1.48,  1.14,  1.16,  1.15,  1.43,  1.43,  1.22])
    #yer = np.array([ 1.395,  0.1  ,  0.275,  0.12 ,  0.235,  0.135,  0.155])
    x = c_lmass[np.where(np.logical_and(c_flag>0.9,c_lmass>10.20))]/c_re_kpc[np.where(np.logical_and(c_flag>0.9,c_lmass>10.20))]**2
    y = c_dk[np.where(np.logical_and(c_flag>0.9,c_lmass>10.20))]
    yer = c_dk_err[np.where(np.logical_and(c_flag>0.9,c_lmass>10.20))]
    sort = np.argsort(x)
    x = x[sort]
    x = np.array(x)
    y = y[sort] 
    y = np.array(y)
    yer = yer[sort]
    yer = np.array(yer)
    xer=zeros(len(x))
    cov=zeros(len(x))   # no correlation between error measurements
    i=0 
    nboot = 10000
    a,b,erra,errb,covab=bces.bces.bcesp(x,xer,y,yer,cov,nboot)
    #a = np.array([ -2.01088498e-02,  -3.34308960e+01,   1.28590319e-01,3.38432765e-02])
    #b = np.array([  1.50112872e+00,   3.56210472e+02,  -7.18932240e-02,9.29291802e-01])
    ybces=a[3]*x+b[3]  # the integer corresponds to the desired BCES method for plotting (3-ort, 0-y|x, 1-x|y, *don't use bissector*)
        # array with best-fit parameters
    fitm=np.array([ a[i],b[i] ])
        # covariance matrix of parameter uncertainties
    covm=np.array([ (aerr[i]**2,covab[i]), (covab[i],berr[i]**2) ])
        # Gets lower and upper bounds on the confidence band 
    lcb1,ucb1,x=nmmn.stats.confband(x, y, a[i], b[i], 0.68, x)
    lcb2,ucb2,x2=nmmn.stats.confband(x, y, a[i], b[i], 0.95, x)
    lcb3,ucb3,x3=nmmn.stats.confband(x, y, a[i], b[i], 0.997, x)
    errorbar(x,y,yerr=yer,fmt='o')
    ax = plot(x,ybces,'-k')
    ax = fill_between(x, lcb1, ucb1, alpha=0.6, facecolor='purple')
    ax = fill_between(x, lcb2, ucb2, alpha=0.3, facecolor='blue')
    ax = fill_between(x, lcb3, ucb3, alpha=0.4, facecolor='grey')
    ax = xlabel('IRC0218 LogMass/R$^2$ SF Mass Complete')
    ax = ylabel('D4000')
    
    
    ax = subplot(2,2,4)
    #ax = ylim(0,2.5)
    x = c_lmass[np.where(np.logical_and(c_flag<0.9,c_lmass>10.20))]/c_re_kpc[np.where(np.logical_and(c_flag<0.9,c_lmass>10.20))]**2
    y = c_dk[np.where(np.logical_and(c_flag<0.9,c_lmass>10.20))]
    yer = c_dk_err[np.where(np.logical_and(c_flag<0.9,c_lmass>10.20))]
    sort = np.argsort(x)
    x = x[sort]
    x = np.array(x)
    y = y[sort] 
    y = np.array(y)
    yer = yer[sort]
    yer = np.array(yer)
    xer=zeros(len(x))
    cov=zeros(len(x))   # no correlation between error measurements
    i = 0
    nboot=10000   # number of bootstrapping trials
    def func(x): return x[1]*x[0]+x[2]
    a,b,aerr,berr,covab=bces.bces.bcesp(x,xer,y,yer,cov,nboot)
    ybces=a[3]*x+b[3]  # the integer corresponds to the desired BCES method for plotting (3-ort, 0-y|x, 1-x|y, *don't use bissector*)
    # array with best-fit parameters
    fitm=np.array([ a[i],b[i] ])
    # covariance matrix of parameter uncertainties
    covm=np.array([ (aerr[i]**2,covab[i]), (covab[i],berr[i]**2) ])
    # Gets lower and upper bounds on the confidence band 
    lcb1,ucb1,x=nmmn.stats.confband(x, y, a[i], b[i], 0.68, x)
    lcb2,ucb2,x2=nmmn.stats.confband(x, y, a[i], b[i], 0.95, x)
    lcb3,ucb3,x3=nmmn.stats.confband(x, y, a[i], b[i], 0.997, x)
    errorbar(x,y,yerr=yer,fmt='o')
    ax = plot(x,ybces,'-k')
    ax = fill_between(x, lcb1, ucb1, alpha=0.6, facecolor='purple')
    ax = fill_between(x, lcb2, ucb2, alpha=0.3, facecolor='blue')
    ax = fill_between(x, lcb3, ucb3, alpha=0.4, facecolor='grey')
    ax = xlabel('IRC0218 LogMass/R$^2$ Q Mass Complete')
    ax = ylabel('D4000')    
    
    plt.show()


def ircmr3(x,y,yer,lcb1,lcb2,lcb3,ucb1,ucb2,ucb3):
    ax = sns.set(style="white", color_codes=True)
    ax = plt.figure(figsize=(18, 16), dpi= 80, facecolor='none', edgecolor='k')
    ax = subplot(2,2,1)
    #ax = ylim(0,2.5)
    x = c_lmass[np.where(c_flag>0.9)]/c_re_kpc[np.where(c_flag>0.9)]**3
    y = c_dk[np.where(c_flag>0.9)]
    yer = c_dk_err[np.where(c_flag>0.9)]
    sort = np.argsort(x)
    x = x[sort]
    x = np.array(x)
    y = y[sort] 
    y = np.array(y)
    yer = yer[sort]
    yer = np.array(yer)
    xer=zeros(len(x))
    cov=zeros(len(x))   # no correlation between error measurements
    i = 0
    nboot=10000   # number of bootstrapping trials
    def func(x): return x[1]*x[0]+x[2]
    a,b,aerr,berr,covab=bces.bces.bcesp(x,xer,y,yer,cov,nboot)
    ybces=a[3]*x+b[3]  # the integer corresponds to the desired BCES method for plotting (3-ort, 0-y|x, 1-x|y, *don't use bissector*)
    # array with best-fit parameters
    fitm=np.array([ a[i],b[i] ])
    # covariance matrix of parameter uncertainties
    covm=np.array([ (aerr[i]**2,covab[i]), (covab[i],berr[i]**2) ])
    # Gets lower and upper bounds on the confidence band 
    lcb1,ucb1,x=nmmn.stats.confband(x, y, a[i], b[i], 0.68, x)
    lcb2,ucb2,x2=nmmn.stats.confband(x, y, a[i], b[i], 0.95, x)
    lcb3,ucb3,x3=nmmn.stats.confband(x, y, a[i], b[i], 0.997, x)
    errorbar(x,y,yerr=yer,fmt='o')
    ax = plot(x,ybces,'-k')
    ax = fill_between(x, lcb1, ucb1, alpha=0.6, facecolor='purple')
    ax = fill_between(x, lcb2, ucb2, alpha=0.3, facecolor='blue')
    ax = fill_between(x, lcb3, ucb3, alpha=0.4, facecolor='grey')
    ax = xlabel('IRC0218 LogMass/R$^3$ SF')
    ax = ylabel('D4000')
    
    
    ax = subplot(2,2,2)
    #ax = ylim(0,2.5)
    x = c_lmass[np.where(c_flag<0.9)]/c_re_kpc[np.where(c_flag<0.9)]**3
    y = c_dk[np.where(c_flag<0.9)]
    yer = c_dk_err[np.where(c_flag<0.9)]
    sort = np.argsort(x)
    x = x[sort]
    x = np.array(x)
    y = y[sort] 
    y = np.array(y)
    yer = yer[sort]
    yer = np.array(yer)
    xer=zeros(len(x))
    cov=zeros(len(x))   # no correlation between error measurements
    i = 0
    nboot=10000   # number of bootstrapping trials
    def func(x): return x[1]*x[0]+x[2]
    a,b,aerr,berr,covab=bces.bces.bcesp(x,xer,y,yer,cov,nboot)
    ybces=a[3]*x+b[3]  # the integer corresponds to the desired BCES method for plotting (3-ort, 0-y|x, 1-x|y, *don't use bissector*)
    # array with best-fit parameters
    fitm=np.array([ a[i],b[i] ])
    # covariance matrix of parameter uncertainties
    covm=np.array([ (aerr[i]**2,covab[i]), (covab[i],berr[i]**2) ])
    # Gets lower and upper bounds on the confidence band 
    lcb1,ucb1,x=nmmn.stats.confband(x, y, a[i], b[i], 0.68, x)
    lcb2,ucb2,x2=nmmn.stats.confband(x, y, a[i], b[i], 0.95, x)
    lcb3,ucb3,x3=nmmn.stats.confband(x, y, a[i], b[i], 0.997, x)
    errorbar(x,y,yerr=yer,fmt='o')
    ax = plot(x,ybces,'-k')
    ax = fill_between(x, lcb1, ucb1, alpha=0.6, facecolor='purple')
    ax = fill_between(x, lcb2, ucb2, alpha=0.3, facecolor='blue')
    ax = fill_between(x, lcb3, ucb3, alpha=0.4, facecolor='grey')
    ax = xlabel('IRC0218 LogMass/R$^3$ Q')
    ax = ylabel('D4000')
    
    ax = subplot(2,2,3)
    #x = np.array([  0.0630552 ,   0.13834198,   0.15097889,   1.50982026,    2.83197654,   6.79663764,  10.44563709])
    #y = np.array([ 1.48,  1.14,  1.16,  1.15,  1.43,  1.43,  1.22])
    #yer = np.array([ 1.395,  0.1  ,  0.275,  0.12 ,  0.235,  0.135,  0.155])
    x = c_lmass[np.where(np.logical_and(c_flag>0.9,c_lmass>10.20))]/c_re_kpc[np.where(np.logical_and(c_flag>0.9,c_lmass>10.20))]**3
    y = c_dk[np.where(np.logical_and(c_flag>0.9,c_lmass>10.20))]
    yer = c_dk_err[np.where(np.logical_and(c_flag>0.9,c_lmass>10.20))]
    sort = np.argsort(x)
    x = x[sort]
    x = np.array(x)
    y = y[sort] 
    y = np.array(y)
    yer = yer[sort]
    yer = np.array(yer)
    xer=zeros(len(x))
    cov=zeros(len(x))   # no correlation between error measurements
    i=0 
    nboot = 10000
    a,b,erra,errb,covab=bces.bces.bcesp(x,xer,y,yer,cov,nboot)
    #a = np.array([ -2.01088498e-02,  -3.34308960e+01,   1.28590319e-01,3.38432765e-02])
    #b = np.array([  1.50112872e+00,   3.56210472e+02,  -7.18932240e-02,9.29291802e-01])
    ybces=a[3]*x+b[3]  # the integer corresponds to the desired BCES method for plotting (3-ort, 0-y|x, 1-x|y, *don't use bissector*)
        # array with best-fit parameters
    fitm=np.array([ a[i],b[i] ])
        # covariance matrix of parameter uncertainties
    covm=np.array([ (aerr[i]**2,covab[i]), (covab[i],berr[i]**2) ])
        # Gets lower and upper bounds on the confidence band 
    lcb1,ucb1,x=nmmn.stats.confband(x, y, a[i], b[i], 0.68, x)
    lcb2,ucb2,x2=nmmn.stats.confband(x, y, a[i], b[i], 0.95, x)
    lcb3,ucb3,x3=nmmn.stats.confband(x, y, a[i], b[i], 0.997, x)
    errorbar(x,y,yerr=yer,fmt='o')
    ax = plot(x,ybces,'-k')
    ax = fill_between(x, lcb1, ucb1, alpha=0.6, facecolor='purple')
    ax = fill_between(x, lcb2, ucb2, alpha=0.3, facecolor='blue')
    ax = fill_between(x, lcb3, ucb3, alpha=0.4, facecolor='grey')
    ax = xlabel('IRC0218 LogMass/R$^3$ SF Mass Complete')
    ax = ylabel('D4000')
    
    
    ax = subplot(2,2,4)
    #ax = ylim(0,2.5)
    x = c_lmass[np.where(np.logical_and(c_flag<0.9,c_lmass>10.20))]/c_re_kpc[np.where(np.logical_and(c_flag<0.9,c_lmass>10.20))]**3
    y = c_dk[np.where(np.logical_and(c_flag<0.9,c_lmass>10.20))]
    yer = c_dk_err[np.where(np.logical_and(c_flag<0.9,c_lmass>10.20))]
    sort = np.argsort(x)
    x = x[sort]
    x = np.array(x)
    y = y[sort] 
    y = np.array(y)
    yer = yer[sort]
    yer = np.array(yer)
    xer=zeros(len(x))
    cov=zeros(len(x))   # no correlation between error measurements
    i = 0
    nboot=10000   # number of bootstrapping trials
    def func(x): return x[1]*x[0]+x[2]
    a,b,aerr,berr,covab=bces.bces.bcesp(x,xer,y,yer,cov,nboot)
    ybces=a[3]*x+b[3]  # the integer corresponds to the desired BCES method for plotting (3-ort, 0-y|x, 1-x|y, *don't use bissector*)
    # array with best-fit parameters
    fitm=np.array([ a[i],b[i] ])
    # covariance matrix of parameter uncertainties
    covm=np.array([ (aerr[i]**2,covab[i]), (covab[i],berr[i]**2) ])
    # Gets lower and upper bounds on the confidence band 
    lcb1,ucb1,x=nmmn.stats.confband(x, y, a[i], b[i], 0.68, x)
    lcb2,ucb2,x2=nmmn.stats.confband(x, y, a[i], b[i], 0.95, x)
    lcb3,ucb3,x3=nmmn.stats.confband(x, y, a[i], b[i], 0.997, x)
    errorbar(x,y,yerr=yer,fmt='o')
    ax = plot(x,ybces,'-k')
    ax = fill_between(x, lcb1, ucb1, alpha=0.6, facecolor='purple')
    ax = fill_between(x, lcb2, ucb2, alpha=0.3, facecolor='blue')
    ax = fill_between(x, lcb3, ucb3, alpha=0.4, facecolor='grey')
    ax = xlabel('IRC0218 LogMass/R$^3$ Q Mass Complete')
    ax = ylabel('D4000')    
    
    plt.show()



