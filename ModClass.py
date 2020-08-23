
#==============================================================================
# Created on Feb 10, 2020
#
# @author: Ramin Mehdizad Tekiyeh

# This code is written in Python 3.7.4 , Spyder 3.3.6
#==============================================================================


#==============================================================================
# This module contains all the Classes that are used in the main code
#==============================================================================


#==============================================================================
# importing standard classes
#==============================================================================
import logging
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os


#==============================================================================
# importing module codes
#==============================================================================
import ModVar


#==============================================================================
# defining class for SVM classifier
#==============================================================================
class MLModelClass:
    # this class creates machin learningclassifier and controls it
    
    # initializing class instance parameters
    def __init__(self,datafile):
        self.df=pd.read_csv(datafile)
        self.SVCModel=SVC()
    
    # splitting data into train, validationand test    
    def DataSplit(self,coeff1,coeff2):
        
        # extracting values and target of database
        self.X_data=np.array(self.df[['x1','x2']])
        self.y_data=np.array(self.df['y'])
        
        self.X_trn,self.X_tst,self.y_trn,self.y_tst=train_test_split(
                self.X_data,self.y_data,test_size=coeff1,random_state=42)
        
        # Next, we further partition (X_trn, y_trn) into training and validation sets
        self.X_trn,self.X_val,self.y_trn,self.y_val=train_test_split(
                self.X_trn,self.y_trn,test_size=coeff2,random_state=42)
    
    # in this method, we calculater accuracy score of Part A    
    def AccScor_A(self,C,cv1):
        y_pred=ModVar.SVCTrainedModelsDict_C[C].predict(self.X_trn)

        ModVar.trnErr_A[C]=accuracy_score(self.y_trn,y_pred, normalize=True)
        ModVar.valErr_A[C]=np.mean((cross_val_score(
                ModVar.SVCTrainedModelsDict_C[C], self.X_val,self.y_val,cv=cv1)))
        
    # in this method, we calculater accuracy score of Part B    
    def AccScor_B(self,Gamma,cv1):
        y_pred=ModVar.SVCTrainedModelsDict_Gamma[Gamma].predict(self.X_trn)

        ModVar.trnErr_B[Gamma]=accuracy_score(self.y_trn,y_pred, normalize=True)
        ModVar.valErr_B[Gamma]=np.mean((cross_val_score(
                ModVar.SVCTrainedModelsDict_Gamma[Gamma],self.X_val,self.y_val,cv=cv1)))
    

#==============================================================================
# this class defines logging events and results into *.log file
#        
# Note:
#     All the methods and logging data are created in the methods of this class
#     Then the logging action is done in the main code
#==============================================================================
class LogClass():
    global logger,filehandler
    
    # initializing class instance parameters
    def __init__(self):
#        self.resultspath=''   # initialize
        self.logger=logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.formatter=logging.Formatter('%(message)s')
        self.filehandler=logging.FileHandler(
                            ModVar.resultssubpath+'\\SVM.log')
        self.filehandler.setFormatter(self.formatter)
        self.logger.addHandler(self.filehandler)
        self.splitterLen=84
        self.splitterChar='*'
        self.EndSep=self.splitterChar*self.splitterLen
    
    # this method logs joining of all visualise processes
    def Analysis_Execution_Report(self,n):    
        title=' MONITORING MAIN CODE RUN '
        sp=self.splitterChar*round((self.splitterLen-len(title))/2)
        self.logger.info(sp+title+sp)
        self.logger.info('')
    
    # this method logs joining of all visualise processes
    def AllVisJoined(self,n):    
        self.LogFrmt(n)
        self.logger.info('')
        self.logger.info('All visual threads joined') 
    
    # this method logs joining of accuracy processes   
    def AllAccJoined(self,n):    
        self.LogFrmt(n)
        self.logger.info('')
        self.logger.info('All accuracy threads joined')
        
    # this method logs saving created data into .csv file
    def DataSavdToCSV(self,n):  
        self.LogFrmt(n)
        self.logger.info("Raw data saved into csv file")
    
    # this method logs saving of results and figures    
    def FigResSaved(self,n):
        self.LogFrmt(n)
        self.logger.info('Figures and results successfully saved.')
        self.logger.info(self.EndSep)

    # this method performs the format of logging for each log action    
    def LogFrmt(self,n):
        if n=='M':
            self.formatter=logging.Formatter(' %(message)s')
            self.filehandler.setFormatter(self.formatter)
        elif n=='LM':
            self.formatter=logging.Formatter('%(levelname)s: %(message)s')
            self.filehandler.setFormatter(self.formatter)
        elif n=='TLM':
            self.formatter=logging.Formatter('%(acstime)s: %(levelname)s: %(message)s')
            self.filehandler.setFormatter(self.formatter)
    
    # multi process of Part A start   
    def Model_i_started_A(self,i,n):   
        self.LogFrmt(n)
        self.logger.info(f'Part A : SVM model {i} started'.format(i))
    
    # multi process of Part B start   
    def Model_i_started_B(self,i,n):   
        self.LogFrmt(n)
        self.logger.info(f'Part B : SVM model {i} started'.format(i))    

    # this method logs ParsedData
    def ParsedData(self,n):
        self.LogFrmt(n)
        title=' Data Entered by User '
        sp=self.splitterChar*(round((self.splitterLen-len(title))/2))
        self.logger.info(sp+title+sp)
        self.logger.info('')
        self.logger.info('  Main Script address  ='+ ModVar.args.ScriptPath)
        self.logger.info('  .csv file address    ='+ ModVar.args.DataBasePath)
        self.logger.info('  Work Directory       ='+ ModVar.args.WorkDir)
        # results path is ceated ans set in CreateAndSetResultsDir()
        self.logger.info('  Results Path         ='+ os.getcwd())
        self.logger.info('  Save Plots           ='+ ModVar.args.SavePlot)
        self.logger.info('  Create Log File      ='+ ModVar.args.logFile)
        self.logger.info(self.EndSep)
    
    # this method logs start of the main
    def ProgStart(self,n):
        self.LogFrmt(n)
        self.logger.info('')
        title=' Main Program Started '
        sp=self.splitterChar*(round((self.splitterLen-len(title))/2))
        self.logger.info(sp+title+sp)
        self.logger.info('')
        self.logger.info('')
     
    # this method logs     
    def ProbDef(self,n):
        self.LogFrmt(n)
        self.logger.info('')
        self.logger.info('')
        title=' Problem Definition '
        sp=self.splitterChar*round((self.splitterLen-len(title))/2)
        self.logger.info(sp+title+sp)
        self.logger.info('The problem has two parts:\n')  
        self.logger.info('    Part A: Gamma is constant and C changes')
        self.logger.info('    Part B: C is constant and Gamma changes')
        self.logger.info('')
        self.logger.info('Train percentage: {} '.format(ModVar.tFrac))
        self.logger.info('validation percentage: {}'.format(ModVar.vFrac))
        self.logger.info('Kernel used: {} '.format(ModVar.kernel))
        self.logger.info('C: {} '.format(ModVar.C_PartB))
        self.logger.info('gamma: {} '.format(ModVar.GAMMA_PartA))
        self.logger.info('CrossValidation Group: {} '.format(ModVar.cvNum))
        self.logger.info('')
        self.logger.info('Iteration for different C values:')
        self.logger.info('                   {}'.format(ModVar.C_values))
        self.logger.info('')
        self.logger.info('Iteration for different Gamma values:')
        self.logger.info('                   {}'.format(ModVar.GAMMA_values))
        self.logger.info(self.EndSep)
        self.logger.info('')
        self.logger.info('')
    
    # this method logs writing of results in the format of sentence        
    def ResultsInfo_SentenceForm(self,dt1,dt2,dt3,dt4,dt5,dt6,n):    
        self.LogFrmt(n)
        self.logger.info('')
        self.logger.info('')
        title=' ANALYSIS RESULTS '
        sp=self.splitterChar*round((self.splitterLen-len(title))/2)
        self.logger.info(sp+title+sp)
        self.logger.info('')
        
        self.logger.info('Part A:')
        self.logger.info('')
        self.logger.info(f'   Elapsed time of run for different C is:     {dt1} s'.format(dt1))
        self.logger.info(f'   Elapsed time of Train\CrossVal is:     {dt2} s'.format(dt2))
        self.logger.info(f'   Total elapsed time of part A is:     {dt3} s'.format(dt3))
        self.logger.info('')
        
        self.logger.info('Train errors and crossValidations are:\n')  
        for i,[C,TrnEr] in enumerate(ModVar.trnErrArray_A):
            self.logger.info('   For C = {} : Train Error is: {} % and Cross Validation Error is: {} %'.format(
                    C,round(10000*TrnEr)/100,round(10000*ModVar.valErrArray_A[i][1])/100))
        
        self.logger.info('')
        self.logger.info('')
        self.logger.info('Part B:')
        self.logger.info('')
        self.logger.info(f'   Elapsed time of run for different Gamma is:     {dt4} s'.format(dt4))
        self.logger.info(f'   Elapsed time of Train\CrossVal is:     {dt5} s'.format(dt5))
        self.logger.info(f'   Total elapsed time of part B is:     {dt6} s'.format(dt6))
        self.logger.info('')
        
        self.logger.info('Train errors and crossValidations are:\n')  
        for i,[G,TrnEr] in enumerate(ModVar.trnErrArray_B):
            self.logger.info('   For Gamma = {} : Train Error is: {} % and Cross Validation Error is: {} %'.format(
                    G,round(10000*TrnEr)/100,round(10000*ModVar.valErrArray_B[i][1])/100))
        
        self.logger.info(self.EndSep)
        
    # this method logs writing of results in the format of table    
    def ResultsInfo_TableForm(self,dt1,dt2,dt3,dt4,dt5,dt6,n):    
        # printing title
        self.LogFrmt(n)
        self.logger.info('')
        self.logger.info('')
        title=' RESULTS IN TABLE FORM '
        sp=self.splitterChar*round((self.splitterLen-len(title))/2)
        self.logger.info(sp+title+sp)
        self.logger.info('')
            
        self.logger.info('======Run Time Table:')
        self.logger.info('')
        
        #  elapsed times table headers
        hd1='run for C/Gamma'
        hd2='Train\CrossVal'
        hd3='Total time'
        Row_title=['Part A ','Part B ']
        
        # header start point 1,2,3
        hd1_start=15
        hd2_start=40
        hd3_start=60
        # space 1,2,3
        spc1=hd1_start * ' '
        spc2=(hd2_start-len(hd1)-len(spc1))* ' '
        spc3=(hd3_start-len(hd1)-len(hd2)-len(spc1)-len(spc2))* ' '
        # print header
        self.logger.info(spc1+hd1+spc2+hd2+spc3+hd3)
        self.logger.info('')
        
        dt_list=[[dt1,dt2,dt3],[dt4,dt5,dt6]]
        for i,[i1,i2,i3] in enumerate(dt_list):
            hd1=str(i1)
            hd2=str(i2)
            hd3=str(i3)
            TabIndent='| '
            
            spc1=(hd1_start-len(TabIndent)-len(Row_title[i])) * ' ' 
            spc1=spc1+Row_title[i]+TabIndent
            
            spc2=((hd2_start-len(hd1)-len(spc1))-len(TabIndent))* ' ' 
            spc2=spc2+TabIndent
            
            spc3=(hd3_start-len(hd1)-len(hd2)-len(spc1)-len(spc2)-len(TabIndent))* ' '
            spc3=spc3+TabIndent
            
            self.logger.info(spc1+hd1+spc2+hd2+spc3+hd3)
        self.logger.info('')    
        self.logger.info('')    
            

        # print accuracy table of part A
        # accuracy headers 1,2,3
        self.logger.info('======Accuracy Table Part A:')
        self.logger.info('')
        hd1='C'   
        hd2='Train Error (%)'
        hd3='Cross-Val Error (%)'
        # header start point 1,2,3
        hd1_start=10
        hd2_start=25
        hd3_start=50
        # space 1,2,3
        spc1=hd1_start * ' '
        spc2=(hd2_start-len(hd1)-len(spc1))* ' '
        spc3=(hd3_start-len(hd1)-len(hd2)-len(spc1)-len(spc2))* ' '
        # print header
        self.logger.info(spc1+hd1+spc2+hd2+spc3+hd3)
        self.logger.info('')
            
        for i,[C,TrnEr] in enumerate(ModVar.trnErrArray_A):
            hd1=str(C)
            hd2=str(round(10000*TrnEr)/100)
            hd3=str(round(10000*ModVar.valErrArray_A[i][1])/100)
            TabIndent='| '
            
            spc1=(hd1_start-len(TabIndent)) * ' ' 
            spc1=spc1+TabIndent
            
            spc2=((hd2_start-len(hd1)-len(spc1))-len(TabIndent))* ' ' 
            spc2=spc2+TabIndent
            
            spc3=(hd3_start-len(hd1)-len(hd2)-len(spc1)-len(spc2)-len(TabIndent))* ' '
            spc3=spc3+TabIndent
            
            self.logger.info(spc1+hd1+spc2+hd2+spc3+hd3)
        self.logger.info('')
        self.logger.info('')
        
        
        # print accuracy table of part B
        # accuracy headers 1,2,3
        self.logger.info('======Accuracy Table Part B:')
        self.logger.info('')
        hd1='Gamma'   
        hd2='Train Error (%)'
        hd3='Cross-Val Error (%)'
        # header start point 1,2,3
        hd1_start=10
        hd2_start=25
        hd3_start=50
        # space 1,2,3
        spc1=hd1_start * ' '
        spc2=(hd2_start-len(hd1)-len(spc1))* ' '
        spc3=(hd3_start-len(hd1)-len(hd2)-len(spc1)-len(spc2))* ' '
        # print header
        self.logger.info(spc1+hd1+spc2+hd2+spc3+hd3)
        self.logger.info('')
            
        for i,[G,TrnEr] in enumerate(ModVar.trnErrArray_B):
            hd1=str(G)
            hd2=str(round(10000*TrnEr)/100)
            hd3=str(round(10000*ModVar.valErrArray_B[i][1])/100)
            TabIndent='| '
            
            spc1=(hd1_start-len(TabIndent)) * ' ' 
            spc1=spc1+TabIndent
            
            spc2=((hd2_start-len(hd1)-len(spc1))-len(TabIndent))* ' ' 
            spc2=spc2+TabIndent
            
            spc3=(hd3_start-len(hd1)-len(hd2)-len(spc1)-len(spc2)-len(TabIndent))* ' '
            spc3=spc3+TabIndent
            
            self.logger.info(spc1+hd1+spc2+hd2+spc3+hd3)
        
        self.logger.info('')
        self.logger.info(self.EndSep)


    # this method logs the system on which the analysis if performed  
    def SysSpec(self,sysinfo,n):
        self.LogFrmt(n)
        self.logger.info('')
        self.logger.info('')
        title=' COMPUTER SPEC '
        sp=self.splitterChar*round((self.splitterLen-len(title))/2)
        self.logger.info(sp+title+sp)
        self.logger.info('Data analsys is done on the system with following spec:\n')  
        for i,[a1,a2] in enumerate(ModVar.sysinfo):
            DataStartChar=30
            len1=len(ModVar.sysinfo[i][0])
            Arrow='-'*(DataStartChar-len1)+'> '
            self.logger.info(ModVar.sysinfo[i][0]+Arrow+ModVar.sysinfo[i][1])
        self.logger.info(self.EndSep)

    # this method logs the creation of subplot structure
    def SubPlotStruc_PartA(self,n):
        self.LogFrmt(n)
        title=' Part A Started '
        Ind=10*' '
        Aster=10*'*'
        self.logger.info(Ind+Aster+title+Aster)
        self.logger.info('')
        self.logger.info("SubPlot Structure Part A Created\n")
        
    # this method logs the creation of subplot structure
    def SubPlotStruc_PartB(self,n):
        self.LogFrmt(n)
        title=' Part B Started '
        Ind=10*' '
        Aster=10*'*'
        self.logger.info(Ind+Aster+title+Aster)
        self.logger.info('')
        self.logger.info("SubPlot Structure Part B Created\n")
    
    # this method logs the start of timer for parallel running of visualize func  
    def TimeVisStrt(self,n): 
        self.LogFrmt(n)
        self.logger.info("timer for visual threads started \n")
    
    # this method logs the start of timer for parallel running of accuracy func     
    def TimeAccStrt(self,n):
        self.LogFrmt(n)
        self.logger.info("timer for accuracy-validation started \n")
    
    # this method logs finishing time of accuracy processes par A   
    def TimeAccFin(self,n):    
        self.LogFrmt(n)
        self.logger.info('Accuracy\CrossVal done successfully !!!')
        self.logger.info('')

    # this method logs finish time of visualize function    
    def TimeVisFin(self,dt,n):    
        self.LogFrmt(n)
        self.logger.info('Visualize calculations done successfully !!!')
        self.logger.info('')

    # this method logs start ofa specific cross validation process    
    def Train_CrossVal_i_started_PartA(self,i,n):   
        self.LogFrmt(n)
        self.logger.info(f'Part A: Train\CrossVal {i} started'.format(i))
        
    # this method logs start ofa specific cross validation process    
    def Train_CrossVal_i_started_PartB(self,i,n):   
        self.LogFrmt(n)
        self.logger.info(f'Part B: Train\CrossVal {i} started'.format(i))

       # this method logs plotting the input data set    
    def XYPlotted(self,n): 
        self.LogFrmt(n)
        self.logger.info("Raw data plotted successfully")
        self.logger.info('')










