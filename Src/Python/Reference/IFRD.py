#!/home/locker/EE/mscad/david/local/bin/python
'''
Created on Jul 17, 2014

@author: DavidHosoonShin
'''

import os
import numpy as np
import scipy as sp

from scipy import stats, optimize
from scipy.stats import pareto, norm

import matplotlib.pyplot as plt
import datetime
import time
import copy

from REclass import *

if __name__ == '__main__':

    scaleFactor = 1.0e+10
    seed=5
    figN=1
    
    m=27
    nRef=600000
    n0=2000
    n=n0
    sigma=5
    
    criFactor=0.97
    
    refCriFactor1=0.99977 # 5-sigma level
    criFactor1=0.9991
    
    redDim=27
    redSize=1000
    
    # active learning sampling param
    eliteRatio=0.20
    filterType = ['fromMin', 'fromMax', 'both']
    
    path = "/home/locker/EE/mscad/david/REscope/SRAM/SIM/"
    #refResultFile = path + "6t_sram_monte.out"

    # file writing
    now=time.localtime()
    nowTime=str(now.tm_mon)+'_'+str(now.tm_mday)+'_' \
            +str(now.tm_hour)+':'+str(now.tm_min)+':'+str(now.tm_sec)
    filename = 'IFRD_'+ nowTime
    fns = 'IFRD_SPEC_'+ nowTime
    f=open(filename,'w')
    fs = open(fns,'w')
    f.write( "## Conditions##\n")
    f.write( "-Criteria = %s\n" % criFactor)
    f.write( "-Reference # of sample = %s\n" % nRef)
    f.write( "-# of pre-sample = %s\n" % n)
    f.write( "-Dimension = %s\n" % m)
    f.write("\n")

    # Step 0.  reference data processing
    #refResultFile = path + "6t_sram_ref.out"

    print "*Reference processing start"
    #refResultFile = path + "6t_sram_normal_"+str(nRef)+".out"
    refResultFile = path + "6t_sram_normal_600000.out"
    ref=REsim(nRef,m, sigma)
    ref.makeReference(refResultFile)
    
    '''
    # getting [MIN, MAX] of process parameters
    print ref.paramMin
    print ref.paramMax
    exit(1)
    '''

    '''
    sim = REsim (nRef,m,sigma)
    ref=REsample(m,nRef, 'normal') 
    ref.makeData(sim,seed) 
    outputFile = path + "logic_noraml_1000000.out"
    sim.runSampleSim(ref, outputFile)
    '''
    tcFit=REfit(scaleFactor, refCriFactor1)
    tcFit.fitting(ref) #t3 = tcFit.failVal

    
    ## Step 1. Pre sample building and do simulation (SOBOL, RANDOM)
    print "*Pre-sampling processing start"

    '''
    nom=REsample(m,n, 'normal') 
    nom.makeData(ref, seed) 
    outputFile = path + "6t_sram_nom.out_"+str(n)+"_seed5"
    ref.runSampleSim(nom,  outputFile)
    nom=makeSample(nom,0)
    nom.calWeight()
    '''
    '''
    outputFile = path + "6t_sram_sob.out_200_seed5"
    sob=REsim(n,m, sigma)
    sob.makeReference(outputFile)
    sob= makeSample(sob,0)
    '''

    outputFile = path + "6t_sram_nom.out_"+str(n)+"_seed5"
    nom=REsim(n,m, sigma)
    nom.makeReference(outputFile)
    nom= makeSample(nom,0)
    nom.calWeight()

    # simulation data processing for building distribution

    nomCandFit=REfit(scaleFactor, criFactor)
    nomCandFit.fitting(nom)
    
    ## Step 3. Re-evaluation for likely-to-fail samples
    # 1st iteration 
    print "START 1st iteration"
    f.write ("\n1st iteration\n")
    n=n*30
    
    '''
    grbfNom = REcf('grbf', scaleFactor)
    grbfNom.learn(nom)

    cand=REsample(m,n, 'normal') 
    cand.makeData(ref, seed)
    
    # do classification
    grbfNom.classify(cand)

    # filter likely-to-fail sample and do simulation
    grbfSample = makeSample(grbfNom,nomCandFit.failVal / scaleFactor)
    grbfSample.calWeight()
    f.write("- elitism correlation  : %s\n" % grbfSample.corr)
    grbfSampleBoth=makeEliteSample(grbfSample,eliteRatio, filterType[2])

    f.write( "# of fail samples (1st grbf) : %s \n " % grbfSample.size)
    f.write( "# of fail samples (1st grbf_both) : %s \n "% grbfSampleBoth.size)

    grbfFailOutputFile = path + "6t_sram_1stfail_grbf.out_5sigma"+str(n)
    ref.runSampleSim(grbfSample, grbfFailOutputFile)
    outputFile = path + "6t_sram_1stfail_grbf_both.out_5sigma"+str(n)+"_"+str(eliteRatio)
    ref.runSampleSim(grbfSampleBoth, outputFile)

    '''
    # make sample by simulated data
    outputFile = path + "6t_sram_1stfail_grbf.out_5sigma60000"
    grbfSample=REsim(1342,m, sigma)
    grbfSample.makeReference(outputFile)
    grbfSample=makeSample(grbfSample, 0)

    outputFile = path + "6t_sram_1stfail_grbf_both.out_5sigma60000_0.2"
    grbfSampleBoth=REsim(268,m, sigma)
    grbfSampleBoth.makeReference(outputFile)
    grbfSampleBoth=makeSample(grbfSampleBoth, 0)

    f.write( "# of fail samples (1st grbf) : %s \n " % grbfSample.size)
    f.write( "# of fail samples (1st grbf_both) : %s \n "% grbfSampleBoth.size)

    '''
    ######## Start : Spec Validation ###############
    # Dimension reduction based on correlation coefficient
    # # of in-spec check : 10000
    # # of real sample check : 10000*10

    # 1st : cross section spec validation (1 by 1, min and max)
    redDim=27
    [specList, argIdx, cf] = specValidation1(ref, grbfSample, redDim, refFit.failVal / scaleFactor, redSize, scaleFactor)
    print "Speclist 1 (all consideration)"
    fs.write( "Speclist 1 (all consideration)\n")
    for item in specList.items(): 
        print  item 
        fs.write("%s\n" % str(item))
    print "# of spec check:",len(specList)

    # testing for in-spec-data
    testSample = REsample(grbfSample.dim, redSize*10,'sobol') 
    testSample.makePrunDataBySpecList(ref, specList, len(specList), specList.keys(), seed=5) 
    
    cf.classifySpec(testSample)
    testSampleFail = makeSample(testSample, refFit.failVal / scaleFactor)
    
    print "# of testSample: ", len(testSample.param)
    print "# of testSampleFail : ", len(testSampleFail.param) 
    fs.write( "# of testSample: %s\n "% len(testSample.param))
    fs.write( "# of testSampleFail : %s\n "% len(testSampleFail.param)) 

    # 2nd : intersection of all-condition (CC order based, binary decision tree)
    redDim=5
    [test2Sample, argIdx, cf] = specValidation2(ref, grbfSample, redDim, refFit.failVal / scaleFactor, redSize*10, scaleFactor)
    
    cf.classifySpec(test2Sample)
    test2SampleFail = makeSample(test2Sample, refFit.failVal / scaleFactor)

    print "Speclist 2 (using CC order)"
    fs.write( "Speclist 2 (using CC order)\n")
    specList={}
    for idx in argIdx:
        specList[idx]=[np.min(test2Sample.param[:,idx]), np.max(test2Sample.param[:,idx])]
    for item in specList.items(): 
        print  item 
        fs.write("%s\n" % str(item))
    print "# of spec check:",len(specList)

    # testing for in-spec-data
    testSample = REsample(grbfSample.dim, redSize*10,'sobol') 
    testSample.makePrunDataBySpecList(ref, specList, len(specList), specList.keys(), seed=5) 
    
    cf.classifySpec(testSample)
    testSampleFail = makeSample(testSample, refFit.failVal / scaleFactor)
    
    print "# of testSample: ", len(testSample.param)
    print "# of testSampleFail : ", len(testSampleFail.param)
    fs.write( "# of testSample: %s\n "% len(testSample.param))
    fs.write( "# of testSampleFail : %s\n "% len(testSampleFail.param)) 

    # 3rd : intersection of all-condition (variance based)
    redDim=5
    [test2Sample , argIdx, cf]= specValidation3(ref, grbfSample, redDim, refFit.failVal / scaleFactor, redSize*10, scaleFactor)
    
    cf.classifySpec(test2Sample)
    test2SampleFail= makeSample(test2Sample, refFit.failVal / scaleFactor)
    print "Speclist 3 (using variance order)"
    fs.write( "Speclist 3 (using variance order)\n")
    specList={}
    for idx in argIdx:
        specList[idx]=[np.min(test2Sample.param[:,idx]), np.max(test2Sample.param[:,idx])]
    for item in specList.items(): 
        print  item 
        fs.write("%s\n" % str(item))
    print "# of spec check:",len(specList)

    # testing for in-spec-data
    testSample = REsample(grbfSample.dim, redSize*10,'sobol') 
    testSample.makePrunDataBySpecList(ref, specList, len(specList), specList.keys(), seed=5) 
    
    cf.classifySpec(testSample)
    testSampleFail = makeSample(testSample, refFit.failVal / scaleFactor)
    
    print "# of testSample: ", len(testSample.param)
    print "# of testSampleFail : ", len(testSampleFail.param)
    fs.write( "# of testSample: %s\n "% len(testSample.param))
    fs.write( "# of testSampleFail : %s\n "% len(testSampleFail.param)) 

    # 4th : intersection of all-condition (Tree-based feature selection)
    redDim=5
    [test2Sample , argIdx, cf]= specValidation4(ref, grbfSample, redDim, refFit.failVal / scaleFactor, redSize*10, scaleFactor)
    
    cf.classifySpec(test2Sample)
    test2SampleFail= makeSample(test2Sample, refFit.failVal / scaleFactor)
    print "Speclist 4 (using Tree-based feature selection)"
    fs.write( "Speclist 4 (using Tree-based feature selection)\n")
    specList={}
    for idx in argIdx:
        specList[idx]=[np.min(test2Sample.param[:,idx]), np.max(test2Sample.param[:,idx])]
    for item in specList.items(): 
        print  item 
        fs.write("%s\n" % str(item))
    print "# of spec check:",len(specList)

    # testing for in-spec-data
    testSample = REsample(grbfSample.dim, redSize*10,'sobol') 
    testSample.makePrunDataBySpecList(ref, specList, len(specList), specList.keys(), seed=5) 
    
    cf.classifySpec(testSample)
    testSampleFail = makeSample(testSample, refFit.failVal / scaleFactor)
    
    print "# of testSample: ", len(testSample.param)
    print "# of testSampleFail : ", len(testSampleFail.param)
    fs.write( "# of testSample: %s\n "% len(testSample.param))
    fs.write( "# of testSampleFail : %s\n "% len(testSampleFail.param)) 
    ######## End : Spec Validation ###############
    '''

    # fitting
    grbfSample1st = makeSample(grbfSample, nomCandFit.failVal / scaleFactor)
    grbfSample1stW = weightedSample(grbfSample1st,4)
    grbfSample1stBoth = makeSample(grbfSampleBoth, nomCandFit.failVal / scaleFactor)
    grbfSample1stBothW = weightedSample(grbfSample1stBoth,8)

    grbfFit1st=REfit(scaleFactor, criFactor)
    grbfFit1st.fittingNew(grbfSample1stW, 1)
    grbfFit1stBoth=REfit(scaleFactor, criFactor)
    grbfFit1stBoth.fittingNew(grbfSample1stBothW, 1)

    grbfPIS1ANorm = criFactor +(1-criFactor)*grbfFit1st.findNormCDF(tcFit.failVal)
    grbfPIS1ALog = criFactor +(1-criFactor)*grbfFit1st.findLogCDF(tcFit.failVal)
    grbfPIS1APareto = criFactor +(1-criFactor)*grbfFit1st.findParetoCDF(tcFit.failVal)
    grbfPIS1ANormBoth = criFactor +(1-criFactor)*grbfFit1stBoth.findNormCDF(tcFit.failVal)
    grbfPIS1ALogBoth = criFactor +(1-criFactor)*grbfFit1stBoth.findLogCDF(tcFit.failVal)
    grbfPIS1AParetoBoth = criFactor +(1-criFactor)*grbfFit1stBoth.findParetoCDF(tcFit.failVal)

    ## 2nd iteration 
    # make nCand sample
    grbfSample=makeNSample(grbfSample,n0)

    print "START 2nd iteration"
    f.write ("\n2nd iteration\n")
    n=n*10

    '''

    cand=REsample(m,n, 'normal') 
    cand.makeData(ref, seed)

    # build 2nd classifier using 1st tc
    grbf2nd = REcf('grbf', scaleFactor)
    grbf2nd.learn(grbfSample)
    grbf2nd.classify(cand)

    # filter likely-to-fail sample and do simulation
    grbfSample2nd = makeSample(grbf2nd,grbfFit1st.failValPareto / scaleFactor)
    grbfSample2ndBoth = makeSample(grbf2nd,grbfFit1st.failValPareto / scaleFactor)
    grbfSample2ndBoth.calWeight()
    f.write("- elitism correlation  : %s\n" % grbfSample2ndBoth.corr)
    grbfSample2ndBoth=makeEliteSample(grbfSample2ndBoth,eliteRatio, filterType[2])
    
    f.write( "# of fail samples (2nd grbf) : %s\n " % grbfSample2nd.size)
    f.write( "# of fail samples (2nd grbf_both) : %s\n "% grbfSample2ndBoth.size)

    outputFile = path + "6t_sram_2ndfail_grbf.out_5sigma"+str(n)
    ref.runSampleSim(grbfSample2nd, outputFile)
    outputFile = path + "6t_sram_2ndfail_grbf_both.out_5sigma"+str(n)+str(eliteRatio)
    ref.runSampleSim(grbfSample2ndBoth, outputFile)

    '''
    # make sample by simulated data
    outputFile = path + "6t_sram_2ndfail_grbf.out_5sigma600000"
    grbfSample2nd=REsim(6961,m, sigma)
    grbfSample2nd.makeReference(outputFile)
    grbfSample2nd=makeSample(grbfSample2nd, 0)
    
    outputFile = path + "6t_sram_2ndfail_grbf_both.out_5sigma6000000.2"
    grbfSample2ndBoth=REsim(1392,m, sigma)
    grbfSample2ndBoth.makeReference(outputFile)
    grbfSample2ndBoth=makeSample(grbfSample2ndBoth, 0)

    f.write( "# of fail samples (2nd grbf) : %s\n " % grbfSample2nd.size)
    f.write( "# of fail samples (2nd grbf_both) : %s\n "% grbfSample2ndBoth.size)
    
    grbfSample2nd = makeNSample(grbfSample2nd, 3000)
    grbfSample2nd.calWeight()

    ######## Start : Spec Validation ###############
    # Dimension reduction based on correlation coefficient
    # # of in-spec check : 10000
    # # of real sample check : 10000*10

    '''
    # 1st : cross section spec validation (1 by 1, min and max)
    redDim=27
    [specList, argIdx, cf] = specValidation1(ref, grbfSample2nd, redDim, tcFit.failVal / scaleFactor, redSize, scaleFactor)
    print "Speclist 1 (all consideration)"
    fs.write( "Speclist 1 (all consideration)\n")
    for item in specList.items(): 
        print  item 
        fs.write("%s\n" % str(item))
    print "# of spec check:",len(specList)

    # testing for in-spec-data
    testSample = REsample(grbfSample2nd.dim, redSize*10,'sobol') 
    testSample.makePrunDataBySpecList(ref, specList, len(specList), specList.keys(), seed=5) 
    
    cf.classifySpec(testSample)
    testSampleFail = makeSample(testSample, tcFit.failVal / scaleFactor)
    
    print "# of testSample: ", len(testSample.param)
    print "# of testSampleFail : ", len(testSampleFail.param) 
    fs.write( "# of testSample: %s\n "% len(testSample.param))
    fs.write( "# of testSampleFail : %s\n "% len(testSampleFail.param)) 

    # 2nd : intersection of all-condition (CC order based, binary decision tree)
    redDim=5
    [test2Sample, argIdx, cf] = specValidation2(ref, grbfSample2nd, redDim, tcFit.failVal / scaleFactor, redSize*10, scaleFactor)
    
    cf.classifySpec(test2Sample)
    test2SampleFail = makeSample(test2Sample, tcFit.failVal / scaleFactor)

    print "Speclist 2 (using CC order)"
    fs.write( "Speclist 2 (using CC order)\n")
    specList={}
    for idx in argIdx:
        specList[idx]=[np.min(test2Sample.param[:,idx]), np.max(test2Sample.param[:,idx])]
    for item in specList.items(): 
        print  item 
        fs.write("%s\n" % str(item))
    print "# of spec check:",len(specList)

    # testing for in-spec-data
    testSample = REsample(grbfSample2nd.dim, redSize*10,'sobol') 
    testSample.makePrunDataBySpecList(ref, specList, len(specList), specList.keys(), seed=5) 
    
    cf.classifySpec(testSample)
    testSampleFail = makeSample(testSample, tcFit.failVal / scaleFactor)
    
    print "# of testSample: ", len(testSample.param)
    print "# of testSampleFail : ", len(testSampleFail.param)
    fs.write( "# of testSample: %s\n "% len(testSample.param))
    fs.write( "# of testSampleFail : %s\n "% len(testSampleFail.param)) 

    # 3rd : intersection of all-condition (variance based)
    redDim=5
    [test2Sample , argIdx, cf]= specValidation3(ref, grbfSample2nd, redDim, tcFit.failVal / scaleFactor, redSize*10, scaleFactor)
    
    cf.classifySpec(test2Sample)
    test2SampleFail= makeSample(test2Sample, tcFit.failVal / scaleFactor)
    print "Speclist 3 (using variance order)"
    fs.write( "Speclist 3 (using variance order)\n")
    specList={}
    for idx in argIdx:
        specList[idx]=[np.min(test2Sample.param[:,idx]), np.max(test2Sample.param[:,idx])]
    for item in specList.items(): 
        print  item 
        fs.write("%s\n" % str(item))
    print "# of spec check:",len(specList)

    # testing for in-spec-data
    testSample = REsample(grbfSample2nd.dim, redSize*10,'sobol') 
    testSample.makePrunDataBySpecList(ref, specList, len(specList), specList.keys(), seed=5) 
    
    cf.classifySpec(testSample)
    testSampleFail = makeSample(testSample, tcFit.failVal / scaleFactor)
    
    print "# of testSample: ", len(testSample.param)
    print "# of testSampleFail : ", len(testSampleFail.param)
    fs.write( "# of testSample: %s\n "% len(testSample.param))
    fs.write( "# of testSampleFail : %s\n "% len(testSampleFail.param)) 

    # 4th : intersection of all-condition (Tree-based feature selection)
    redDim=5
    [test2Sample , argIdx, cf]= specValidation4(ref, grbfSample2nd, redDim, tcFit.failVal / scaleFactor, redSize*10, scaleFactor)
    
    cf.classifySpec(test2Sample)
    test2SampleFail= makeSample(test2Sample, tcFit.failVal / scaleFactor)
    print "Speclist 4 (using Tree-based feature selection)"
    fs.write( "Speclist 4 (using Tree-based feature selection)\n")
    specList={}
    for idx in argIdx:
        specList[idx]=[np.min(test2Sample.param[:,idx]), np.max(test2Sample.param[:,idx])]
    for item in specList.items(): 
        print  item 
        fs.write("%s\n" % str(item))
    print "# of spec check:",len(specList)

    # testing for in-spec-data
    testSample = REsample(grbfSample2nd.dim, redSize*10,'sobol') 
    testSample.makePrunDataBySpecList(ref, specList, len(specList), specList.keys(), seed=5) 
    
    cf.classifySpec(testSample)
    testSampleFail = makeSample(testSample, tcFit.failVal / scaleFactor)
    
    print "# of testSample: ", len(testSample.param)
    print "# of testSampleFail : ", len(testSampleFail.param)
    fs.write( "# of testSample: %s\n "% len(testSample.param))
    fs.write( "# of testSampleFail : %s\n "% len(testSampleFail.param)) 
    ######## End : Spec Validation ###############
    '''

    grbfSample2nd = makeSample(grbfSample2nd, grbfFit1st.failValPareto / scaleFactor)
    grbfSample2ndW = weightedSample(grbfSample2nd,3.5)
    grbfSample2ndBoth = makeSample(grbfSample2ndBoth, grbfFit1stBoth.failValPareto / scaleFactor)
    grbfSample2ndBothW = weightedSample(grbfSample2ndBoth,6)

    grbfFit2nd=REfit(scaleFactor,criFactor)
    grbfFit2nd.fittingNew(grbfSample2ndW, 2)
    grbfFit2ndBoth=REfit(scaleFactor,criFactor)
    grbfFit2ndBoth.fittingNew(grbfSample2ndBothW, 2)

    grbfPIS2Norm = criFactor1 +(1-criFactor1)*grbfFit2nd.findNormCDF(tcFit.failVal)
    grbfPIS2Log = criFactor1 +(1-criFactor1)*grbfFit2nd.findLogCDF(tcFit.failVal)
    grbfPIS2Pareto = criFactor1 +(1-criFactor1)*grbfFit2nd.findParetoCDF(tcFit.failVal)
    grbfPIS2NormBoth = criFactor1 +(1-criFactor1)*grbfFit2ndBoth.findNormCDF(tcFit.failVal)
    grbfPIS2LogBoth = criFactor1 +(1-criFactor1)*grbfFit2ndBoth.findLogCDF(tcFit.failVal)
    grbfPIS2ParetoBoth = criFactor1 +(1-criFactor1)*grbfFit2ndBoth.findParetoCDF(tcFit.failVal)

    # 2nd fail distribution plotting

    plt.figure(figN)
    plt.title('SRAM READ GPD (2nd)')
    plt.plot(tcFit.normX,tcFit.normCDF,'k-') 
    plt.plot(grbfFit1st.normX,grbfFit1st.paretoCDF,'k--') 
    plt.plot(grbfFit2nd.normX,grbfFit2nd.paretoCDF,'k-.') 
    plt.plot(grbfFit2ndBoth.normX,grbfFit2ndBoth.paretoCDF,'k:') 
    plt.legend(['MC', 'REscope', 'RSB', 'IFRD'], 'best' ) 
    plt.grid(True)
    figN+=1

    f.write( "## 2nd re-run statistics##\n")
    f.write(" REF : %s\n" % (1- tcFit.findNormCDF(tcFit.failVal)))
    f.write(" RSB_Norm 2nd NEW : %s\n" % grbfPIS2Norm)
    f.write(" RSB_Log 2nd NEW : %s\n" % grbfPIS2Log)
    f.write(" RSB_Pareto 2nd NEW : %s\n" % grbfPIS2Pareto)
    f.write(" IFRD_Norm 2nd NEW : %s\n" % grbfPIS2NormBoth)
    f.write(" IFRD_Log 2nd NEW : %s\n" % grbfPIS2LogBoth)
    f.write(" IFRD_Pareto 2nd NEW : %s\n" % grbfPIS2ParetoBoth)
    f.write( "## Approximated by 1st  ##\n")
    f.write(" RESCOPE_Norm approximated 2nd NEW : %s\n" % grbfPIS1ANorm)
    f.write(" RESCOPE_Log approximated 2nd NEW : %s\n" % grbfPIS1ALog)
    f.write(" RESCOPE_Pareto approximated 2nd NEW : %s\n" % grbfPIS1APareto)
    
    plt.show()
    f.close()
    fs.close()
