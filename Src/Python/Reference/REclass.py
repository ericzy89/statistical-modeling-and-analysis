'''
Created on Jun 25, 2014

@author: DavidHosoonShin
'''

import os
import numpy as np
import scipy as sp

import datetime
import time
import copy
from copy import deepcopy

from sobol_seq import *

from scipy.interpolate import Rbf
from scipy import stats, optimize
from scipy.stats import pareto, norm, lognorm
from numpy import linalg 

#import pygsl
#import mlpy as mp 

from pybrain.tools.shortcuts import buildNetwork 
from pybrain.datasets import SupervisedDataSet 
from pybrain.datasets import ClassificationDataSet 
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.supervised.trainers import RPropMinusTrainer
from pybrain.structure import TanhLayer
from pybrain.structure import SoftmaxLayer

from sklearn import svm
from sklearn.decomposition import PCA
#from matplotlib.mlab import PCA
from sklearn.decomposition import KernelPCA
from sklearn.ensemble import ExtraTreesClassifier

class REsample:
    '''
    - pre sampling class using SOBOL and RANDOM
    '''
    dim=None
    size=None
    
    param=None
    result=None
    type=None
    max=None
    min=None
    corr=None
    
    weightFromMin=None
    weightFromMax=None
    weightFromMed=None
    weightToMed=None

    weightMinIdx=None
    weightMaxIdx=None

    normFactor=np.array([ 1.0e+1, 1.0e+9, 1.0e+02, 1.0e-18, 1.0e+09, 1.0e+09, 1.0e-00, 1.0e+10, \
                1.0e+10, 1.0e+01, 1.0e+09, 1.0e+02, 1.0e-18, 1.0e+09, 1.0e+09, 1.0e-00, \
                1.0e+10, 1.0e+10, 1.0e+01, 1.0e+09, 1.0e+03, 1.0e-18, 1.0e+09, 1.0e+09,\
                1.0e-00, 1.0e+10, 1.0e+10 ])

    def __init__(self, dim, size, type=None):
        '''
        Constructor
        '''
        self.dim=dim
        self.size=size
        self.type=type
        
    '''
    def makeData(self, seed=0): 
        if self.type is 'sobol': 
            self.param=i4_sobol_generate ( self.dim, self.size, seed ) 
            self.param=np.transpose(self.param)
        elif self.type is 'rand': 
            np.random.seed(seed) 
            self.param=np.random.rand ( self.dim, self.size) 
            self.param=np.transpose(self.param)
        elif self.type is 'normal': 
            np.random.seed(seed) 
            self.param=np.random.randn (uself.dim, self.size) 
            self.param=np.transpose(self.param)
    '''

    def makeData(self, sim,  seed=0): 
        if self.type is 'sobol': 
            self.param=i4_sobol_generate ( self.dim, self.size, seed ) 
            self.param=np.transpose(self.param) 
            self.param = sim.paramInit + np.abs(sim.paramInit) * ((2*self.param-1)*sim.paramInitStd*sim.sigma) 
        elif self.type is 'rand': 
            np.random.seed(seed) 
            self.param=np.random.rand ( self.dim, self.size) 
            self.param=np.transpose(self.param)
            self.param = sim.paramInit + np.abs(sim.paramInit) * ((2*self.param-1)*sim.paramInitStd*sim.sigma) 
        elif self.type is 'normal': 
            np.random.seed(seed) 
            self.param=np.random.randn ( self.dim, self.size) 
            self.param=np.transpose(self.param)
            self.param = sim.paramInit + np.abs(sim.paramInit) * (self.param*sim.paramInitStd) 

        self.max=np.max(self.param,0)
        self.min=np.min(self.param,0)
        
    def makePrunData(self, sim, orgSample, redDim, argIdx, seed=0 ):
        self.param=np.zeros([self.dim, self.size])

        if self.type is 'sobol': 
            pv=i4_sobol_generate ( redDim, self.size, seed ) 
            pvIdx=0
            for dimIdx in range(self.dim): 
                if dimIdx in argIdx: 
                    '''
                    paramInit=np.median(orgSample.param[:,dimIdx])
                    paramStd=np.std(orgSample.param[:,dimIdx])
                    self.param[dimIdx] = paramInit + np.abs(paramInit) * ((2*pv[pvIdx]-1)*paramStd*sim.sigma) 
                    '''
                    diff=np.max(orgSample.param[:,dimIdx])-np.min(orgSample.param[:,dimIdx])
                    self.param[dimIdx]=np.min(orgSample.param[:,dimIdx])+diff*pv[pvIdx]
                    pvIdx+=1
                else:
                    self.param[dimIdx]=sim.paramInit[dimIdx]
            self.param=np.transpose(self.param)
        elif self.type is 'rand': 
            np.random.seed(seed) 
            pv=np.random.rand (redDim, self.size) 
            pvIdx=0
            for dimIdx in range(self.dim): 
                if dimIdx in argIdx: 
                    paramInit=np.median(orgSample.param[:,dimIdx])
                    paramStd=np.std(orgSample.param[:,dimIdx])
                    self.param[dimIdx] = paramInit + np.abs(paramInit) * ((2*pv[pvIdx]-1)*paramStd*sim.sigma) 
                    pvIdx+=1
                else:
                    self.param[dimIdx]=sim.paramInit[dimIdx]
            self.param=np.transpose(self.param)
        elif self.type is 'normal': 
            np.random.seed(seed) 
            pv=np.random.randn ( redDim, self.size) 
            pvIdx=0
            for dimIdx in range(self.dim): 
                if dimIdx in argIdx: 
                    paramInit=np.median(orgSample.param[:,dimIdx])
                    paramStd=np.std(orgSample.param[:,dimIdx])
                    self.param[dimIdx] = paramInit + np.abs(paramInit) * (pv[pvIdx]*paramStd) 
                    pvIdx+=1
                else:
                    self.param[dimIdx]=sim.paramInit[dimIdx]
            self.param=np.transpose(self.param)

        self.max=np.max(self.param,0)
        self.min=np.min(self.param,0)

    def makePrunDataBySpecList(self, sim, specList, redDim, argIdx, seed=0 ):
        self.param=np.zeros([self.dim, self.size])

        if self.type is 'sobol': 
            pv=i4_sobol_generate ( redDim, self.size, seed ) 
            pvIdx=0
            for dimIdx in range(self.dim): 
                if dimIdx in argIdx: 
                    '''
                    paramInit=np.median(orgSample.param[:,dimIdx])
                    paramStd=np.std(orgSample.param[:,dimIdx])
                    self.param[dimIdx] = paramInit + np.abs(paramInit) * ((2*pv[pvIdx]-1)*paramStd*sim.sigma) 
                    '''
                    diff=specList[dimIdx][1]-specList[dimIdx][0]
                    self.param[dimIdx]=specList[dimIdx][0]+diff*pv[pvIdx]
                    pvIdx+=1
                else:
                    self.param[dimIdx]=sim.paramInit[dimIdx]
            self.param=np.transpose(self.param)

        self.max=np.max(self.param,0)
        self.min=np.min(self.param,0)
    
    def costFunc(self, coef): 
        self.result=np.dot(self.param,coef) 
        
    '''
    def calPCA(self): 
        pca = PCA(n_components=self.dim) 
        pca.fit(self.param) 
        print len(self.param)
        print len(pca.components_)
        print pca.components_ 
        print pca.explained_variance_ratio_
        print pca.explained_variance_
    '''

    def calWeight(self):
        self.weightToMin=[]
        self.weightToMax=[]

        # calculate correlation
        self.corr = np.zeros([self.dim])
        for idx in range(self.dim):
            if all(self.param[:,idx]) == 0 or len(np.unique(self.param[:,idx])) ==1 : 
                self.corr[idx]=0
            else:
                tmpCorr = np.corrcoef(self.param[:,idx], self.result) 
                self.corr[idx]=np.abs(tmpCorr[0,1])

        # decide reference sample point 
        minIdx = np.argmin(self.result)
        maxIdx = np.argmax(self.result)
        medIdx = ((self.result).argsort()[::-1])[self.size/2]
        
        self.weightMinIdx=minIdx
        self.weightMaxIdx=maxIdx

        minParam=self.param[minIdx]
        maxParam=self.param[maxIdx]
        medParam=self.param[medIdx]

        #minParam=np.min(self.param, 0)
        #maxParam=np.max(self.param, 0)
        #medParam=np.median(self.param, 0)
        
        normalization = np.abs(np.max(self.param,0)-np.min(self.param,0)) 
        normalization[np.where(normalization == 0.)[0]]=1.0 

        # calculate distance from reference sample point & final weight for each sample
        self.weightFromMin=[]
        self.weightFromMax=[]
        self.weightFromMed=[]
        self.weightToMed=[]

        for idx in range(self.size):
            #distFromMin= np.abs(self.param[minIdx] - self.param[idx]) * self.normFactor
            #distFromMax= np.abs(self.param[maxIdx] - self.param[idx]) * self.normFactor

            # More far from min, max point, More bigger weight
            distFromMin= np.abs(self.param[idx] - minParam) / normalization
            distFromMax= np.abs(self.param[idx] - maxParam) / normalization

            # More far to median point, More bigger weight
            distFromMed= (np.abs(self.param[idx] - medParam) / normalization)
            # More close to median point, More bigger weight
            distToMed=  (np.abs(self.param[idx] - medParam) / normalization) 
            distToMed[np.where(distToMed != 0.)[0]]=1/distToMed[np.where(distToMed != 0.)[0]]
            
            weightFromMinTmp = np.sum(distFromMin * self.corr)
            weightFromMaxTmp = np.sum(distFromMax * self.corr)
            weightFromMedTmp = np.sum(distFromMed * self.corr)
            weightToMedTmp = np.sum(distToMed * self.corr)
            
            self.weightFromMin.append(weightFromMinTmp)
            self.weightFromMax.append(weightFromMaxTmp)
            self.weightFromMed.append(weightFromMedTmp)
            self.weightToMed.append(weightToMedTmp)

        self.weightFromMin=np.asarray(self.weightFromMin)
        self.weightFromMax=np.asarray(self.weightFromMax)
        self.weightFromMed=np.asarray(self.weightFromMed)
        self.weightToMed=np.asarray(self.weightToMed)
        

'''
class REprun:
    #- parameter prunning class 

    # ReliefF algorithm parameter
    iter=2000
    sigma=1.0
    theta=0.0001

    failVal=None
    redDim=None
    weight=None
    argIdx =None

    obj=None
    normFactor=np.array([ 1.0e+1, 1.0e+9, 1.0e+02, 1.0e-18, 1.0e+09, 1.0e+09, 1.0e-00, 1.0e+10, \
                1.0e+10, 1.0e+01, 1.0e+09, 1.0e+02, 1.0e-18, 1.0e+09, 1.0e+09, 1.0e-00, \
                1.0e+10, 1.0e+10, 1.0e+01, 1.0e+09, 1.0e+03, 1.0e-18, 1.0e+09, 1.0e+09,\
                1.0e-00, 1.0e+10, 1.0e+10 ])
    scaleFactor=None

    def __init__(self,redDim, failVal,scaleFactor, sigma ):
        # Constructor

        self.scaleFactor = scaleFactor
        self.failVal=failVal 
        self.redDim=redDim
        self.sigma=sigma
        self.obj = mp.IRelief(self.iter,self.sigma,self.theta)
        
        assert self.redDim is not None and self.failVal is not None, 'Fail : No definition (in REprun)'

    def pruning(self, sample): 

        normResult = sample.result * self.scaleFactor
        normParam = sample.param * self.normFactor
        criY=(normResult>=self.failVal)*1 
        criY=criY.flatten()
        
        if len(np.unique(criY)) ==2 : 
            print "ReliefF learning start"
            self.obj.learn(normParam, criY) 
            self.weight=self.obj.weights() 
            print "ReliefF learning end"
        else: 
            assert False,  "Criteria is unique!!"
        assert self.weight is not None, 'False : weight is None'
        
        self.argIdx = np.abs(self.weight).argsort()[::-1][:self.redDim]
        assert self.argIdx is not None,  'False : argIdx is None'
'''

class REcf:
    '''
    - Package class for various classifier
    - GRBF, KNN, Neural Network
    '''
    # classifier type (grbk, knn, nn)
    type=None

    # normalization factor
    normFactor=np.array([ 1.0e+1, 1.0e+9, 1.0e+02, 1.0e-18, 1.0e+09, 1.0e+09, 1.0e-00, 1.0e+10, \
                1.0e+10, 1.0e+01, 1.0e+09, 1.0e+02, 1.0e-18, 1.0e+09, 1.0e+09, 1.0e-00, \
                1.0e+10, 1.0e+10, 1.0e+01, 1.0e+09, 1.0e+03, 1.0e-18, 1.0e+09, 1.0e+09,\
                1.0e-00, 1.0e+10, 1.0e+10 ])
    scaleFactor=None

    # object for classifier
    obj=None
    
    #variables for NN
    net=None
    ds=None
    
    # variables for classification
    result=None
    param=None
    dim=None
    size=None
    
    def __init__(self, type, scaleFactor ):
        self.type=type
        self.scaleFactor = scaleFactor
        assert self.type=='grbf' or self.type=='knn' or self.type=='nn' , "Fail : Do not choose classifier type" 

    def learn(self,sample): 

        print "Start learning by %s" % self.type 
        normParam = sample.param * self.normFactor
        normResult = sample.result * self.scaleFactor

        '''
        if self.type == 'grbf' : 
            cmd = 'self.obj=Rbf(' 
            for i in range(sample.dim): 
                cmd=cmd+'normParam[:,'+str(i)+'],' 
            cmd=cmd+'normResult, function=\'gaussian\')' 
            exec(cmd) 
            '''
        if self.type == 'grbf' : 
            self.obj=svm.NuSVC()
            self.obj.fit(normParam,normResult)
        elif self.type == 'knn':
            self.obj = mp.KNN(sample.size -1)
            self.obj.learn(normParam, normResult.flatten())
        elif self.type == 'nn':
            self.ds=SupervisedDataSet(sample.dim, 1)
            for idx in range(sample.size):
                self.ds.addSample(normParam[idx], normResult[idx])
            self.net= buildNetwork(self.ds.indim, self.ds.indim+10, self.ds.outdim, bias=True)
            self.obj = BackpropTrainer(self.net,self.ds)
            self.obj.trainUntilConvergence(maxEpochs=100)
        else :
            assert False, 'Fail : No classifier type'

        print "End learning by %s" % self.type 

    def classify(self,sample):
        normParam = sample.param * self.normFactor
        normResult=None

        print "Start classification by %s" % self.type 

        if self.type == 'grbf' : 
            tmpValue=[]
            for item in normParam:
                tmpValue.append(self.obj.predict(item))
            normResult = np.asarray(tmpValue).flatten()
        elif self.type == 'knn':
            normResult=self.obj.pred(normParam)
        elif self.type == 'nn':
            tmpValue=[]
            for item in normParam:
                tmpValue.append(self.net.activate(item))
            normResult = np.asarray(tmpValue).flatten()
        else :
            assert False, 'Fail : No classifier type'

        self.result = normResult / self.scaleFactor
        self.param=sample.param
        self.dim=sample.dim
        self.size=sample.size
        print "End classification by %s" % self.type 

    def classifySpec(self,sample):
        normParam = sample.param * self.normFactor
        normResult=None

        print "Start classification by %s" % self.type 

        if self.type == 'grbf' : 
            tmpValue=[]
            for item in normParam:
                tmpValue.append(self.obj.predict(item))
            normResult = np.asarray(tmpValue).flatten()
        elif self.type == 'knn':
            normResult=self.obj.pred(normParam)
        elif self.type == 'nn':
            tmpValue=[]
            for item in normParam:
                tmpValue.append(self.net.activate(item))
            normResult = np.asarray(tmpValue).flatten()
        else :
            assert False, 'Fail : No classifier type'

        sample.result = normResult / self.scaleFactor
        print "End classification by %s" % self.type 

        
class REfit:
    '''
    - All region :  Fitting data to PDF and CDF of gaussian distribution 
    - Failure region :  Fitting data to PDF and CDF of generalized pareto distribution 
    '''
    
    scaleFactor=None
    criFactor=None
    
    result=None
    failResult=None
    normX=None
    norm=None
    failX=None
    fail=None
    
    failVal=None
    failValPareto=None
    failValLog=None

    failCnt=None
    bin=None
    
    normPDF=None
    normCDF=None
    normCDFSpot=None

    failPDF=None
    failCDF=None
    failMean=None
    failStd=None

    param=None
    paramLog=None
    paramPareto=None
    paramFail=None

    def __init__(self, scaleFactor, criFactor ):
        self.scaleFactor=scaleFactor
        self.criFactor=criFactor
        assert self.scaleFactor is not None and self.criFactor is not None, 'Fail : no definition for criFactor or scaleFactor'
        
    def fitting(self, sample, failVal=None): 
        self.result=np.sort(sample.result.flatten() * self.scaleFactor)
        param = norm.fit(self.result) 
        paramLog = lognorm.fit(self.result) 
        paramPareto = pareto.fit(self.result) 

        self.param=param 
        self.paramLog=paramLog
        self.paramPareto=paramPareto
        
        self.normX=np.linspace(np.min(sample.result)*self.scaleFactor, np.max(sample.result)*self.scaleFactor, sample.size) 
        self.normMean = norm.mean(param[0], param[1])
        self.normStd = norm.std(param[0], param[1])
        self.normCDF=norm.cdf(self.normX,param[0],param[1]) 
        self.normPDF=norm.pdf(self.normX,param[0],param[1]) 

        self.logMean = lognorm.mean(paramLog[0], paramLog[1], paramLog[2])
        self.logStd = lognorm.std(paramLog[0], paramLog[1],paramLog[2])
        self.logCDF=lognorm.cdf(self.normX,paramLog[0],paramLog[1],paramLog[2]) 
        self.logPDF=lognorm.pdf(self.normX,paramLog[0],paramLog[1],paramLog[2]) 

        self.paretoMean = pareto.mean(paramPareto[0], paramPareto[1], paramPareto[2])
        self.paretoStd = pareto.std(paramPareto[0], paramPareto[1], paramPareto[2])
        self.paretoCDF=pareto.cdf(self.normX,paramPareto[0],paramPareto[1], paramPareto[2]) 
        self.paretoPDF=pareto.pdf(self.normX,paramPareto[0],paramPareto[1], paramPareto[2]) 
        
        # Filtering Y > yt and fitting pareto 
        if failVal == None : 
            self.failVal= norm.ppf(self.criFactor, param[0],param[1])
            self.failValPareto= pareto.ppf(self.criFactor, paramPareto[0],paramPareto[1],paramPareto[2])
            self.failValLog= lognorm.ppf(self.criFactor, paramLog[0],paramLog[1], paramLog[2])
        else:
            self.failVal = failVal

        self.failCnt=np.count_nonzero((self.result >= self.failVal)*1) 
        self.bin = (self.result>=self.failVal)*1 
        
        self.failResult = self.result[-self.failCnt:] 
        # Fitting reference fail data to GPD 
        paramFail = pareto.fit(self.failResult) 
        self.paramFail=paramFail
        self.failX=np.linspace(np.min(self.failResult), np.max(self.failResult), self.failCnt) 
        self.failCDF=pareto.cdf(self.failX,paramFail[0],paramFail[1],paramFail[2]) 
        self.failPDF=pareto.pdf(self.failX,paramFail[0],paramFail[1],paramFail[2]) 
        self.failMean = pareto.mean(paramFail[0], paramFail[1], paramFail[2])
        self.failStd = pareto.std(paramFail[0], paramFail[1], paramFail[2])
        self.fail=stats.pareto(self.failMean, self.failStd)

    def fittingNew(self, sample, iterN, failVal=None): 
        self.result=np.sort(sample.result.flatten() * self.scaleFactor)
        param = norm.fit(self.result )
        paramLog = lognorm.fit(self.result)
        paramPareto = pareto.fit(self.result)
        

        self.param=param 
        self.paramLog=paramLog
        self.paramPareto=paramPareto
        
        self.normX=np.linspace(np.min(sample.result)*self.scaleFactor, np.max(sample.result)*self.scaleFactor, sample.size) 
        self.normMean = norm.mean(param[0], param[1])
        self.normStd = norm.std(param[0], param[1])
        self.normCDF=(1.0-pow((1.0-self.criFactor),iterN))+norm.cdf(self.normX,param[0],param[1])*pow((1.0-self.criFactor), iterN)
        self.normPDF=norm.pdf(self.normX,param[0],param[1]) 

        self.logMean = lognorm.mean(paramLog[0], paramLog[1], paramLog[2])
        self.logStd = lognorm.std(paramLog[0], paramLog[1],paramLog[2])
        self.logCDF=(1.0-pow((1.0-self.criFactor),iterN))+lognorm.cdf(self.normX,paramLog[0],paramLog[1],paramLog[2])*pow((1.0-self.criFactor), iterN)
        self.logPDF=lognorm.pdf(self.normX,paramLog[0],paramLog[1],paramLog[2]) 

        self.paretoMean = pareto.mean(paramPareto[0], paramPareto[1], paramPareto[2])
        self.paretoStd = pareto.std(paramPareto[0], paramPareto[1], paramPareto[2])
        self.paretoCDF=(1.0-pow((1.0-self.criFactor), iterN))+pareto.cdf(self.normX,paramPareto[0],paramPareto[1], paramPareto[2])*pow((1.0-self.criFactor), iterN)
        self.paretoPDF=pareto.pdf(self.normX,paramPareto[0],paramPareto[1], paramPareto[2]) 
        
        # Filtering Y > yt and fitting pareto 
        if failVal == None : 
            self.failVal= norm.ppf(self.criFactor, param[0],param[1])
            self.failValPareto= pareto.ppf(self.criFactor, paramPareto[0],paramPareto[1],paramPareto[2])
            self.failValLog= lognorm.ppf(self.criFactor, paramLog[0],paramLog[1], paramLog[2])
            '''
            self.failVal=self.findVal(iterN+1, 'norm')
            self.failValPareto=self.findVal(iterN+1, 'pareto')
            self.failValLog=self.findVal(iterN+1, 'log')
            '''
        else:
            self.failVal = failVal

        self.failCnt=np.count_nonzero((self.result >= self.failVal)*1) 
        self.bin = (self.result>=self.failVal)*1 
        
        self.failResult = self.result[-self.failCnt:] 
        # Fitting reference fail data to GPD 
        paramFail = pareto.fit(self.failResult) 
        self.paramFail=paramFail
        self.failX=np.linspace(np.min(self.failResult), np.max(self.failResult), self.failCnt) 
        self.failCDF=pareto.cdf(self.failX,paramFail[0],paramFail[1],paramFail[2]) 
        self.failPDF=pareto.pdf(self.failX,paramFail[0],paramFail[1],paramFail[2]) 
        self.failMean = pareto.mean(paramFail[0], paramFail[1], paramFail[2])
        self.failStd = pareto.std(paramFail[0], paramFail[1], paramFail[2])
        self.fail=stats.pareto(self.failMean, self.failStd)
        
        
    def findNormCDF(self, value):
        return norm.cdf(value,self.param[0],self.param[1]) 
    def findLogCDF(self, value):
        return lognorm.cdf(value,self.paramLog[0],self.paramLog[1], self.paramLog[2]) 
    def findParetoCDF(self, value):
        return pareto.cdf(value,self.paramPareto[0],self.paramPareto[1], self.paramPareto[2]) 

class REsim:
    '''
    - Reading MC sim. result file and make reference data
    - Building simulation running input deck
    - Doing spice simulation
    '''

    #for building reference data section
    dim=None
    size=None
    sigma=None

    param=None
    paramName=['@nch[vfb]','@nch[toxe]','@nch[u0]','@nch[ndep]','@nch[lint]','@nch[wint]','@nch[rsh]', \
               '@nch[cgso]','@nch[cgdo]','@ncha[vfb]','@ncha[toxe]','@ncha[u0]','@ncha[ndep]','@ncha[lint]', \
               '@ncha[wint]','@ncha[rsh]','@ncha[cgso]','@ncha[cgdo]','@pch[vfb]','@pch[toxe]','@pch[u0]', \
               '@pch[ndep]','@pch[lint]','@pch[wint]','@pch[rsh]','@pch[cgso]','@pch[cgdo]'] 
    
    paramInit=np.array([ -5.5e-01, 1.8e-09, 3.2e-02, 2.8e+18, 1.0e-09, 5.0e-09, 0.0e+00, 6.238e-10, \
                6.238e-10, -5.5e-01, 1.8e-09, 3.2e-02, 2.8e+18, 1.0e-09, 5.0e-09, 0.0e+00, \
                6.238e-10, 6.238e-10, 5.5e-01, 1.8e-09, 9.5e-03, 2.8e+18, 1.0e-09, 5.0e-09,\
                0.0e+00, 6.238e-10, 6.238e-10 ])

    paramMin=None
    paramMax=None
    paramStd=None
    paramAvg=None
    paramInitStd=np.array([0.1,0.05,0.1,0.1,0.05,0.05,0.1,0.1,0.1, \
                            0.1,0.05,0.1,0.1,0.05,0.05,0.1,0.1,0.1, \
                            0.1,0.05,0.1,0.1,0.05,0.05,0.1,0.1,0.1])

    result=None
    resultName=[]
    resultInit=[]
    resultAvg=None
    resultStd=None 
    
    path = "/home/locker/EE/mscad/david/REscope/SRAM/SIM/"
    simRunningFile = path + "6t_sram_run.sp"
    
    def __init__(self, size, dim, sigma):
        '''
        Constructor
        '''
        self.dim=dim
        self.size=size
        self.sigma=sigma
        assert self.dim is not None and self.size is not None and self.sigma is not None, 'Fail : constructing REsim'

    def makeReference(self, filename):
        '''
        Make reference data from MC simulation result
        '''
        fid=open(filename,"r") 
        
        init=False
        #dimIdx=0
        #sizeIdx=0

        dimIdx=0
        sizeIdx=0
        buildInit=True

        for line in fid: 
            if line.strip(): 
                if self.dim is not None and self.size is not None and buildInit is True: 
                    self.param=np.zeros([self.size, self.dim]) 
                    #self.paramInit=np.zeros([self.dim]) 
                    self.result=np.zeros([self.size])
                    buildInit=False

                if line.startswith('@'): 
                    token=line.split()
                    if init is True:
                        self.paramName.append(token[0])
                        self.paramInit[dimIdx]=float(token[2])
                    self.param[sizeIdx][dimIdx]=float(token[2])
                    dimIdx=dimIdx+1

                if line.startswith('readtime_0'):
                    token=line.split()
                    if init is True:
                        self.resultName.append(token[0])
                        self.resultInit.append(float(token[2]))
                    self.result[sizeIdx]=float(token[2])
                    init=False
                    sizeIdx=sizeIdx+1
                    dimIdx=0
                    
                if sizeIdx >= self.size:
                    break
                    
            
        self.paramMin=np.min(self.param,0)
        self.paramMax=np.max(self.param,0)
        self.paramAvg=np.average(self.param, 0)
        self.paramStd=np.std(self.param, 0)
        
        self.resultMin=np.min(self.result)
        self.resultMax=np.max(self.result)
        self.resultAvg=np.average(self.result)
        self.resultStd=np.std(self.result)
        fid.close()

    def runSampleSim(self, sample, outputFile):

        # process variation for simulation running
        assert self.dim == sample.dim, 'Fail : self.dim is not equal to sample.dim'
        #print self.paramInitStd
        #sample.param = self.paramInit + np.abs(self.paramInit) * ((2*sample.param-1)*self.paramInitStd*self.sigma) 
        #print sample.data
        #exit(1)
        fid=open(outputFile,'w')
        fid.write('*** Run begin***\n')
        fid.write('- # of sample : %s\n' % sample.size)
        fid.write('- # of dim : %s\n' % sample.dim)
        fid.close()

        fout= open(outputFile,'a')
        ngspice="/home/locker/EE/mscad/david/local/bin/ngspice -b "
        value=[]
        for idx in range(sample.size): 
            processVar=sample.param[idx]

            self.makeInputDeck(processVar) 

            outputFileTmp = outputFile+"_tmp"
            cmd = ngspice + self.simRunningFile + " -o " + outputFileTmp
            os.system(cmd)
            
            self.makeOutput(outputFileTmp, value)
            ftmp = open(outputFileTmp,'r')
            for line in ftmp:
                fout.write(line)
            ftmp.close()
        
        sample.result = np.asarray(value) 
        fout.close()

    def makeOutput(self,outputFile, value):
        fid=open(outputFile, 'r' ) 
        for line in fid: 
            if line.strip(): 
                if line.startswith('readtime_0'):
                    token=line.split()
                    value.append(float(token[2]))
                    fid.close()
                    break


    def makeInputDeck(self,processVar): 
        fid=open(self.simRunningFile, 'w' ) 
        buf=[] 
            
        # make input deck 
        # prefix part 
        buf.append("* \n") 
        buf.append(" .title SRAM(6t) MC simulation \n") 
        buf.append("*.model nch nmos (version=4.7.0 level=14 ) \n") 
        buf.append("*.model pch pmos (version=4.7.0 level=14 ) \n") 
            
        buf.append(".include '/home/locker/EE/mscad/david/REscope/SRAM/SIM/mos.model' \n") 
        buf.append(".options post scale=1u temp=25 noacct \n") 
        buf.append(".global vdd \n") 
            
        buf.append("* parameters \n") 
        buf.append(".param vp=1.8v \n") 
        buf.append(".param ml=0.3 \n") 
        buf.append(".param wn=0.9 \n") 
        buf.append(".param wp='wn * 3' \n") 
        buf.append(".param wna=0.45 \n") 
        buf.append(".param bitcap=0.1pf \n") 
        buf.append(".param drise  = 400ps \n") 
        buf.append(".param dfall  = 100ps \n") 
        buf.append(".param trise  = 100ps \n") 
        buf.append(".param tfall  = 100ps \n") 
        buf.append(".param period = 1ns \n") 
        buf.append(".param skew_meas = 'vp/2' \n") 
            
        buf.append("* subcir discription \n") 
        buf.append(".subckt inv in out pw='wp' pl='ml' nw='wn' nl='ml' \n") 
        buf.append("mp out in vdd vdd pch w='pw' l='pl' ad='pw*2.5*pl' as='pw*2.5*pl' pd='2*pw+5*pl' ps='2*pw+5*pl' m=1 \n") 
        buf.append("mn out in 0 0 nch w='nw' l='nl' ad='nw*2.5*nl' as='nw*2.5*nl' pd='2*nw+5*nl' ps='2*nw+5*nl' m=1 \n") 
        buf.append(".ends \n") 
            
        buf.append("* circuit discription \n") 
        buf.append("xinvl qb q inv \n") 
        buf.append("xinvr q qb inv \n") 
        buf.append("mnl bl wl q 0 ncha w='wna' l='ml' ad='wna*2.5*ml' as='wna*2.5*ml' pd='2*wna+5*ml' ps='2*wna+5*ml' m=1 \n") 
        buf.append("mnr blb wl qb 0 ncha w='wna' l='ml' ad='wna*2.5*ml' as='wna*2.5*ml' pd='2*wna+5*ml' ps='2*wna+5*ml' m=1 \n") 
        buf.append("cbl bl 0 bitcap \n") 
        buf.append("cblb blb 0 bitcap \n") 
            
        buf.append("* PWL forcing (read) \n") 
        buf.append("vvdd vdd 0 'vp' \n") 
        buf.append("vwl wl 0 pwl 0 0 1n 0 1.1n 'vp' \n") 
        buf.append(".ic v(q)=0 \n") 
        buf.append(".ic v(qb)='vp' \n") 
        buf.append(".ic v(bl)='vp' \n") 
        buf.append(".ic v(blb)='vp' \n") 

        # control part 
        buf.append(".control \n")
        buf.append("set filetype=ascii \n") 
            
        for dimIdx in range(self.dim): 
            buf.append("altermod "+ str(self.paramName[dimIdx]) + "=" + str(processVar[dimIdx]) +" \n")
 
        buf.append("* for setting process variable \n") 
        buf.append("set nchvfb=@nch[vfb] \n") 
        buf.append("set nchtoxe=@nch[toxe] \n") 
        buf.append("set nchu0=@nch[u0] \n") 
        buf.append("set nchndep=@nch[ndep] \n") 
        buf.append("set nchlint=@nch[lint] \n") 
        buf.append("set nchwint=@nch[wint] \n") 
        buf.append("set nchrsh=@nch[rsh] \n") 
        buf.append("set nchcgso=@nch[cgso] \n") 
        buf.append("set nchcgdo=@nch[cgdo] \n") 
        
        buf.append("set pchvfb=@pch[vfb] \n") 
        buf.append("set pchtoxe=@pch[toxe] \n") 
        buf.append("set pchu0=@pch[u0] \n") 
        buf.append("set pchndep=@pch[ndep] \n") 
        buf.append("set pchlint=@pch[lint] \n") 
        buf.append("set pchwint=@pch[wint] \n") 
        buf.append("set pchrsh=@pch[rsh] \n") 
        buf.append("set pchcgso=@pch[cgso] \n") 
        buf.append("set pchcgdo=@pch[cgdo] \n") 
        
        buf.append("set nchavfb=@ncha[vfb] \n") 
        buf.append("set nchatoxe=@ncha[toxe] \n") 
        buf.append("set nchau0=@ncha[u0] \n") 
        buf.append("set nchandep=@ncha[ndep] \n") 
        buf.append("set nchalint=@ncha[lint] \n") 
        buf.append("set nchawint=@ncha[wint] \n") 
        buf.append("set ncharsh=@ncha[rsh] \n") 
        buf.append("set nchacgso=@ncha[cgso] \n") 
        buf.append("set nchacgdo=@ncha[cgdo] \n") 
        
        buf.append("print $nchvfb $nchtoxe $nchu0 $nchndep $nchlint $nchwint $nchrsh $nchcgso $nchcgdo  \n")
        buf.append("+ $nchavfb $nchatoxe $nchau0 $nchandep $nchalint $nchawint $ncharsh $nchacgso $nchacgdo  \n") 
        buf.append("+ $pchvfb $pchtoxe $pchu0 $pchndep $pchlint $pchwint $pchrsh $pchcgso $pchcgdo \n") 
        buf.append("tran 1p 3n \n") 
        buf.append("meas tran readTime_0 trig v(wl) val=1.0 rise=1 targ v(bl) val=1.0 fall=1 \n") 
        buf.append(".endc \n") 
        buf.append(".end \n") 
        
        for line in buf: 
            fid.write(line) 

        fid.close()

def makeSample(obj, failVal):
    # obj type : REsample, REcf
    Param=[]
    Result=[]
    for i in np.argwhere(obj.result >= failVal): 
        Param.append(obj.param[i].flatten()) 
        Result.append(obj.result[i])
    Param = np.asarray(Param)
    Result = np.asarray(Result).flatten()

    Sample = REsample(obj.dim, len(Param)) 

    Sample.param = Param 
    Sample.result=Result
    
    return Sample

def makePSample(obj, failVal):
    # obj type : REsample, REcf
    Param=[]
    Result=[]
    for i in np.argwhere(obj.result < failVal): 
        Param.append(obj.param[i].flatten()) 
        Result.append(obj.result[i])
    Param = np.asarray(Param)
    Result = np.asarray(Result).flatten()

    Sample = REsample(obj.dim, len(Param)) 

    Sample.param = Param 
    Sample.result=Result
    
    return Sample

def makePFSample(obj, failVal):
    # obj type : REsample, REcf
    Param=[]
    Result=[]

    for i in np.argwhere(obj.result >= failVal): 
        Param.append(obj.param[i].flatten()) 
        Result.append(int(0))

    for i in np.argwhere(obj.result < failVal): 
        Param.append(obj.param[i].flatten()) 
        Result.append(int(1))

    Param = np.asarray(Param)
    Result = np.asarray(Result).flatten()

    Sample = REsample(obj.dim, len(Param)) 

    Sample.param = Param 
    Sample.result=Result
    
    return Sample

def weightedSample(obj, weight):
    argIdx = np.abs(obj.result).argsort()[::1]
    print len(argIdx)

    w = np.power(np.linspace(2,1, len(argIdx)), weight)
    w=w.astype(np.int64)

    param=[]
    result=[]

    iidx=0
    for idx in argIdx: 
        for idxw in range(w[iidx]):
            param.append(obj.param[idx].flatten())
            result.append(obj.result[idx])
        iidx=iidx+1

    param = np.asarray(param)
    result = np.asarray(result).flatten()

    sample = REsample(obj.dim, len(param)) 
    sample.param = param 
    sample.result=result
    
    return sample

def makeNSample(obj, n0): 
    # obj type : REsample, REcf
    param=[]
    result=[]

    if obj.size < n0:
        n0=obj.size
        
    sample = REsample(obj.dim, n0) 
    argIdx = np.abs(obj.result).argsort()[::-1][:n0]
    for idx in argIdx: 
            param.append(obj.param[idx].flatten())
            result.append(obj.result[idx])
        
    param = np.asarray(param)
    result = np.asarray(result).flatten()

    sample.param = param 
    sample.result=result
    
    return sample

def expandSample(sample1, sample2): 
    # obj type : REsample
    expandSample = REsample(sample1.dim, sample1.size+sample2.size) 
    expandSample.param=np.concatenate((sample1.param,sample2.param), axis=0) 
    expandSample.result=np.concatenate((sample1.result,sample2.result), axis=0) 
    
    return expandSample

def makeEliteSample(sample, ratio, filterType):
    size = int(ratio * sample.size)
    newSample = REsample(sample.dim, size) 
    
    minArgIdx = (sample.weightFromMin).argsort()[::-1][:size]
    maxArgIdx = (sample.weightFromMax).argsort()[::-1][:size] 
    medArgIdx = (sample.weightFromMed).argsort()[::-1][:size] 

    medArgIdxhalf = (sample.weightFromMed).argsort()[::-1][:int(size/2)] 
    medArgIdxhalfCont = (sample.weightToMed).argsort()[::-1][:int(size/2)] 
    
    param=[]
    result=[]
    if filterType == 'fromMin': 
        for idx in minArgIdx: 
            param.append(sample.param[idx].flatten())
            result.append(sample.result[idx])
    elif filterType =='fromMax':
        for idx in maxArgIdx: 
            param.append(sample.param[idx].flatten())
            result.append(sample.result[idx])
    elif filterType=='both':
        for idx in medArgIdxhalf: 
            param.append(sample.param[idx].flatten())
            result.append(sample.result[idx])
        for idx in medArgIdxhalfCont: 
            param.append(sample.param[idx].flatten())
            result.append(sample.result[idx])
        newSample.size = 2 * int(size/2)
        '''
        for idx in medArgIdx: 
            param.append(sample.param[idx].flatten())
            result.append(sample.result[idx])
        '''

    param = np.asarray(param)
    result = np.asarray(result).flatten()
    newSample.param = param
    newSample.result= result
    
    return newSample

def calMIS (candSample, failSample): 
    failResult=np.sort(failSample.result.flatten())
    candResult=np.sort(candSample.result.flatten())
    
    failRange = np.linspace(1, 0, len(failSample.result))
    candRange = np.linspace(1, 0, len(candSample.result))
    #failRange = candRange[::-1][:len(failResult)]
    
    pMIS = np.dot(failResult,failRange) / np.dot(candResult,candRange)
    return pMIS
    
def findVal(fit, cri): 
    for item in fit.normCDF: 
        if item >= cri: 
            targetV=fit.normX[np.argwhere(fit.normCDF==item)] 
            break
    return float(targetV.flatten())

def specValidation1(ref,orgSample, redDim, failVal, redSize, scaleFactor):
    
    argIdx = (orgSample.corr).argsort()[::-1][:redDim]
    print "effective correlation coefficient:",argIdx

    cf = REcf('grbf', scaleFactor)
    cf.learn(orgSample)
    
    print "Starting spec-in checking"
    
    specList={} 
    
    '''
    sample = REsample(orgSample.dim, redSize,'sobol') 
    sample.makePrunData(ref, orgSample, redDim, argIdx, seed=5) 
    sample.size = len(sample.param)
    '''

    for pidx in argIdx: 
        pidxList=[]
        pidxList.append(pidx)

        sample = REsample(orgSample.dim, redSize,'sobol')
        sample.makePrunData(ref, orgSample, redDim, pidxList, seed=5)
        sample.size = len(sample.param)

        cf.classifySpec(sample)
        sampleFail = makeSample(sample,failVal)
        
        print "Processing with parameter[%s]"% pidx 
        print "Before processing"
        
        print "tmpSample size:", len(sample.param)
        print "sampleFail size:", len(sampleFail.param)

        if len(sampleFail.param) >0 : 
            maxFailSpec = np.max(sampleFail.param[:,pidx]) 
            minFailSpec = np.min(sampleFail.param[:,pidx]) 
            print "max[%s] : %s" % (pidx, maxFailSpec) 
            print "min[%s] : %s" % (pidx, minFailSpec) 
            
            specList[pidx]=[minFailSpec, maxFailSpec]

    return specList, argIdx, cf

def specValidation2(ref,orgSample, redDim, failVal, redSize, scaleFactor):
    
    argIdx = (orgSample.corr).argsort()[::-1][:redDim]
    print "effective correlation coefficient:",argIdx

    cf = REcf('grbf', scaleFactor)
    cf.learn(orgSample)
    
    print "Starting spec-in checking"
    
    sample = REsample(orgSample.dim, redSize,'sobol') 
    sample.makePrunData(ref, orgSample, redDim, argIdx, seed=5) 
    sample.size = len(sample.param) 
    cf.classifySpec(sample)
    

    for pidx in argIdx: 
        newParam=[]
        newResult=[] 
        sampleFail = makeSample(sample,failVal)
        
        print "Processing with parameter[%s]"% pidx 
        print "Before processing"
        
        print "tmpSample size:", len(sample.param)
        print "sampleFail size:", len(sampleFail.param)

        if len(sampleFail.param) >0 : 
            maxFailSpec = np.max(sampleFail.param[:,pidx]) 
            minFailSpec = np.min(sampleFail.param[:,pidx]) 
            print "max[%s] : %s" % (pidx, maxFailSpec) 
            print "min[%s] : %s" % (pidx, minFailSpec) 
            
            for idx in range(sample.size):
                if sample.param[idx,pidx] < minFailSpec or sample.param[idx,pidx] > maxFailSpec:
                    newParam.append(sample.param[idx])
                    newResult.append(sample.result[idx]) 
                    
            newParam = np.asarray(newParam) 
            newResult = np.asarray(newResult).flatten() 
            
            sample.param = newParam 
            sample.result= newResult
            sample.size=len(newParam)

    return sample, argIdx, cf

def specValidation3(ref,orgSample, redDim, failVal, redSize, scaleFactor): 
    
    var=np.var(orgSample.param,0)
    argIdx = (var).argsort()[::-1][:redDim] 
    print "effective correlation coefficient:",argIdx

    cf = REcf('grbf', scaleFactor)
    cf.learn(orgSample)
    
    print "Starting spec-in checking"
    
    sample = REsample(orgSample.dim, redSize,'sobol') 
    sample.makePrunData(ref, orgSample, redDim, argIdx, seed=5) 
    sample.size = len(sample.param) 
    cf.classifySpec(sample)
    

    for pidx in argIdx: 
        newParam=[]
        newResult=[] 
        sampleFail = makeSample(sample,failVal)
        
        print "Processing with parameter[%s]"% pidx 
        print "Before processing"
        
        print "tmpSample size:", len(sample.param)
        print "sampleFail size:", len(sampleFail.param)

        if len(sampleFail.param) >0 : 
            maxFailSpec = np.max(sampleFail.param[:,pidx]) 
            minFailSpec = np.min(sampleFail.param[:,pidx]) 
            print "max[%s] : %s" % (pidx, maxFailSpec) 
            print "min[%s] : %s" % (pidx, minFailSpec) 
            
            for idx in range(sample.size):
                if sample.param[idx,pidx] < minFailSpec or sample.param[idx,pidx] > maxFailSpec:
                    newParam.append(sample.param[idx])
                    newResult.append(sample.result[idx]) 
                    
            newParam = np.asarray(newParam) 
            newResult = np.asarray(newResult).flatten() 
            
            sample.param = newParam 
            sample.result= newResult
            sample.size=len(newParam)

    return sample, argIdx, cf

def specValidation4(ref,orgSample, redDim, failVal, redSize, scaleFactor): 
    
    clf = ExtraTreesClassifier() 
    clf.fit(orgSample.param, orgSample.result)
    argIdx = (clf.feature_importances_).argsort()[::-1][:redDim] 
    print "effective correlation coefficient:",argIdx

    cf = REcf('grbf', scaleFactor)
    cf.learn(orgSample)
    
    print "Starting spec-in checking"
    
    sample = REsample(orgSample.dim, redSize,'sobol') 
    sample.makePrunData(ref, orgSample, redDim, argIdx, seed=5) 
    sample.size = len(sample.param) 
    cf.classifySpec(sample)
    

    for pidx in argIdx: 
        newParam=[]
        newResult=[] 
        sampleFail = makeSample(sample,failVal)
        
        print "Processing with parameter[%s]"% pidx 
        print "Before processing"
        
        print "tmpSample size:", len(sample.param)
        print "sampleFail size:", len(sampleFail.param)

        if len(sampleFail.param) >0 : 
            maxFailSpec = np.max(sampleFail.param[:,pidx]) 
            minFailSpec = np.min(sampleFail.param[:,pidx]) 
            print "max[%s] : %s" % (pidx, maxFailSpec) 
            print "min[%s] : %s" % (pidx, minFailSpec) 
            
            for idx in range(sample.size):
                if sample.param[idx,pidx] < minFailSpec or sample.param[idx,pidx] > maxFailSpec:
                    newParam.append(sample.param[idx])
                    newResult.append(sample.result[idx]) 
                    
            newParam = np.asarray(newParam) 
            newResult = np.asarray(newResult).flatten() 
            
            sample.param = newParam 
            sample.result= newResult
            sample.size=len(newParam)

    return sample, argIdx, cf

################################################
######### Testing classes ######################
################################################

class REex8data:
    '''
    Reading Stanford Univ. 2D-example data for nonlinear learning

    '''
    # GRBF algorithm parameter
    filename=None
    minBound=None
    maxBound=None
    data=None
    dim=None
    value=None
    size=None
    npData=None

    def __init__(self, filename):
        '''
        Constructor
        '''
        self.filename=filename 
        fid=open(filename,"r") 
        self.value=[]
        
        # compress all '+' line to 1 Line
        self.value=[]
        self.data=[]
        for line in fid: 
            if line.strip(): 
                token=line.split()
                self.dim=len(token[1:])
                self.value.append(float(token[0]))
                tmpData=[]
                for dataToken in token[1:]:
                    tmpData.append(float(dataToken.split(":")[1]))
                self.data.append(tmpData)

        self.size=len(self.value)
        '''
        self.npData=np.zeros([self.size,self.dim])
        for i in range(self.size):
            for j in range(self.dim):
                self.npData[i,j]=self.data[i][j]
                
        '''
        #self.data=self.npData

        self.data=np.asarray(self.data)
        self.max=np.max(self.data,0)
        self.min=np.min(self.data,0)
        fid.close()


class REgrbf:
    '''
    - Package class for various classifier
    '''
    # GRBF algorithm parameter
    rbf=None
    value=None
    data=None

    def __init__(self):
        '''
        Constructor
        '''

    def learn(self,sample):
        exeStr = 'self.rbf=Rbf('
        for i in range(sample.dim):
            exeStr=exeStr+'sample.data[:,'+str(i)+'],'
        exeStr=exeStr+'sample.value, function=\'gaussian\')'
        print "REgrbf : start learning"
        exec(exeStr)
        print "REgrbf : end learning"

    def classify(self,sample):
        exeStr = 'self.value=self.rbf('
        for i in range(sample.dim):
            exeStr=exeStr+'sample.data[:,'+str(i)+'],'
        exeStr=exeStr+')'
        print "REgrbf : start classification"
        exec(exeStr)
        self.data=sample.data
        print "REgrbf : end classification"

class RErelief:
    '''
    - parameter prunning class usin GRBF
    '''
    # ReliefF algorithm parameter
    iter=2000
    sigma=1.0
    theta=0.0001
    criteria=None

    rel=None

    def __init__(self,criteria):
        '''
        Constructor
        '''
        self.rel = mp.IRelief(self.iter,self.sigma,self.theta)
        self.criteria=criteria

    def costFunc(self, sample): 
        weight=None

        criY=(sample.result>=self.criteria)*1 
        criY=criY.flatten()
        
        if len(np.unique(criY)) ==2 : 
            print "learning start"
            self.rel.learn(sample.param, criY) 
            weight=self.rel.weights() 
            print "learning end"
        else: 
            print "Criteria is unique!!"
            weight=None
            exit(1)

        return weight
        
