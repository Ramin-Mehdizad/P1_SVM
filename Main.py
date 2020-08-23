

#==============================================================================
# Created on Feb 10, 2020
#
# @author: Ramin Mehdizad Tekiyeh

# This code is written in Python 3.7.4 , Spyder 3.3.6
#==============================================================================


#==============================================================================
# deleting variables before starting main code
#==============================================================================
try:
    from IPython import get_ipython
    get_ipython().magic('reset -sf')
    print('... Variables Deleted ...')
except:
    print('... Couldn"t Delete Variables ...')


#==============================================================================
# importing module codes
#==============================================================================
import ModClass 
import ModVar
import ModFunc


#==============================================================================
# importing standard classes
#==============================================================================
from sklearn.svm import SVC
import time
import os
import threading
import argparse
import sys


#==============================================================================
# main code starts here
#==============================================================================
if __name__=='__main__':
    
    # create parse class
    parser1 = argparse.ArgumentParser(add_help=True,prog='SVM Program',
             description='* This program analyzes DataSet by SVM *')
    
    # set program version
    parser1.add_argument('-v','--version',action='version',
                        version='%(prog)s 2.0')
   
    # script path
    MainDir=os.path.abspath(__file__)
    MainDir=MainDir[0:len(MainDir)-8]
    
    parser1.add_argument('-f', '--ScriptPath', action='store',
                        default=MainDir,
                        dest='ScriptPath',help='Shows Script Address')
    
    # set DataBase file address
    parser1.add_argument('-d', '--dbPath', action='store',
                        default=MainDir+'\DataSet2.csv',
                        dest='DataBasePath',help='Enter .csv file address')
    
    # set work directory
    parser1.add_argument('-w', '--wDirAd', action='store',
                        default=MainDir,
                        dest='WorkDir',help='Enter work Directory')
    
    # whetherto save plots or not
    parser1.add_argument('-p', '--SavePlot', action='store', 
                         default='1',  dest='SavePlot', choices=['0', '1'],
                         help='0: Dont Save plots     1: Save plots')
    
    #whether to create log file or not
    parser1.add_argument('-l', '--log', action='store',
                         default='1', dest='logFile', choices=['0', '1'],
                         help='0: Dont write logfile     1: write logfile')
    
    
    # indicates when to exit while loop
    entry=False
    while entry==False:
        # initialize
        ParsErr=0
        FileErr=0
        makedirErr=0
        
        # --------------in thus section we try to parse successfully-----------
        # function to call input data from command line    
        ModFunc.Input()
        
        # user wanted to continue with default values
        if ModVar.str1=='':
            ModVar.args=parser1.parse_args()
            # exit while loop
            entry=True
        elif ModVar.str1.upper()=='Q':
            # exit script
            sys.exit()
        else:
            entry=True
            ParsErr=0
            try:
                ModVar.args = parser1.parse_args(ModVar.str1.split(' '))
            except:
                entry=False
                ParsErr=1
        #----------------------------------------------------------------------
        
        
        #-------------After having parsed successfully, we coninue-------------
        # continue if parse was done successfully
        if ParsErr==0:  
            #check if data base file exists
            FileErr=0
            if os.path.isfile(ModVar.args.DataBasePath):
                pass
            else:
                print("DataBase file address doesn't exist.")
                print('Enter a valid file address.')
                entry=False
                FileErr=1
            
            # continue of datafile address is correct
            if FileErr==0:
                #check for work dir. if not exist, create it
                if os.path.exists(ModVar.args.WorkDir):
                    pass
                else:
                    makedirErr=0
                    try:
                        # make work dir 
                        os.mkdir(ModVar.args.WorkDir)
                    except OSError:  
                        print("!!!**Work Dir doesn't exist and couldn't be created**!!!")
                        print("Try another directory again")
                        entry=False
                        makedirErr=1
                    except:  
                        print("!!!**Work Dir doesn't exist and couldn't be created**!!!")
                        print("Try another directory again")
                        entry=False
                        makedirErr=1
                    
                    if makedirErr==0:   
                        os.chdir(ModVar.args.WorkDir)
        #----------------------------------------------------------------------
            

    # setting flags for log file andsave plots
    ModFunc.SetFlags()

    # create and set results directory
    ModFunc.CreateAndSetResultsDir()
    
    # print parsed data            
    ModFunc.PrintParsedData()
    
    # create log object
    if ModVar.logFlag: 
        My_Log=ModClass.LogClass()
        # log the data of previous lines
        My_Log.ProgStart('LM')
        My_Log.ParsedData('M')
    
    # logging system information
    ModFunc.GetSysInfo()
    if ModVar.logFlag: My_Log.SysSpec(ModVar.sysinfo,'M')
    
    # problem input data
    ModFunc.ProgParameters()

    # logging problem definition
    if ModVar.logFlag: My_Log.ProbDef('M')
    
    # create an instance of the class
    MLModel_instance=ModClass.MLModelClass(datafile=ModVar.args.DataBasePath)
    MLModel_instance.DataSplit(ModVar.tFrac,ModVar.vFrac)
    
    # plot raw data 
    ModFunc.PlotRawData(MLModel_instance.X_data,MLModel_instance.y_data)
    
    # Analysis Execution Report
    if ModVar.logFlag: My_Log.Analysis_Execution_Report('M')
    if ModVar.logFlag: My_Log.XYPlotted('M')
    
    
    #---------------------part A: analysing C effect---------------------------
    # create sub plot structures
    ModFunc.CreateSubPlotStructure_PartA(MLModel_instance.X_trn)
    if ModVar.logFlag: My_Log.SubPlotStruc_PartA('M')
    
    # starting timer of visualize parallel processing
    RunTimVisA_Start=time.perf_counter()
    if ModVar.logFlag: My_Log.TimeVisStrt('M')
    
    # use threading
    ProcessList2=[]
    for i,C_svc in enumerate(ModVar.C_values):
        
        MLModel_instance.SVCModel=SVC(kernel='rbf',
                    random_state=ModVar.random_state_SVC,gamma=ModVar.GAMMA_PartA,C=C_svc)
        CurrModel=MLModel_instance.SVCModel
        p=threading.Thread(target=ModFunc.visualize_PartA,args=(C_svc,CurrModel,'C',
                    MLModel_instance.X_trn,MLModel_instance.y_trn,i))
        p.start()
        if ModVar.logFlag: My_Log.Model_i_started_A(i,'M')
        ProcessList2.append(p)
        
    for process in ProcessList2:
        process.join()
        
    # log the seuccess ofjoins
    if ModVar.logFlag: My_Log.AllVisJoined('M')
    
    # calculate the finish time     
    RunTimVisA_Finish=time.perf_counter()
    dt1=RunTimVisA_Finish-RunTimVisA_Start
    dt1=round(dt1*100)/100
    
    # log elapsed time
    if ModVar.logFlag: My_Log.TimeVisFin(dt1,'M')
    
    # starting timer of Accuracy\Croo validation parallel processing
    RunTimAccA_Start=time.perf_counter()
    if ModVar.logFlag: My_Log.TimeAccStrt('M')
    
    # use threading
    ProcessList=[]
    for i,C_svc in enumerate(ModVar.C_values):

        p=threading.Thread(target=MLModel_instance.AccScor_A,args=(C_svc,ModVar.cvNum))
        
        p.start()
        if ModVar.logFlag: My_Log.Train_CrossVal_i_started_PartA(i,'M')
        ProcessList.append(p)
    
    # waiting for all Processes to join    
    for process in ProcessList:
        process.join()
        
    # log the seuccess ofjoins
    if ModVar.logFlag: My_Log.AllAccJoined('M')
    
    # calculate the finish time     
    RunTimAccA_Finish=time.perf_counter()
    dt2=RunTimAccA_Finish-RunTimAccA_Start
    dt2=round(dt2*100)/100
    
    # total time
    dt3=dt1+dt2
    
    # log elapsed time
    if ModVar.logFlag: My_Log.TimeAccFin('M')
    
    # plotting train and validation errors
    ModFunc.PltTrnValErr_PartA()
    
    print('trnErrArray_A is:',ModVar.trnErrArray_A)
    print('valErrArray_A is :',ModVar.valErrArray_A)
    #--------------------------------------------------------------------------
    
    
    #---------------------part B: analysing Gamma effect-----------------------
    # create sub plot structures
    ModFunc.CreateSubPlotStructure_PartB(MLModel_instance.X_trn)
    if ModVar.logFlag: My_Log.SubPlotStruc_PartB('M')
    
    # starting timer of visualize parallel processing
    RunTimVisB_Start=time.perf_counter()
    if ModVar.logFlag: My_Log.TimeVisStrt('M')
    
    # use threading
    ProcessList3=[]
    for i,G_svc in enumerate(ModVar.GAMMA_values):
        
        MLModel_instance.SVCModel=SVC(kernel='rbf',
                    random_state=ModVar.random_state_SVC,gamma=G_svc,C=ModVar.C_PartB)
        CurrModel=MLModel_instance.SVCModel
        p=threading.Thread(target=ModFunc.visualize_PartB,args=(G_svc,CurrModel,'Gamma',
                    MLModel_instance.X_trn,MLModel_instance.y_trn,i))
        p.start()
        if ModVar.logFlag: My_Log.Model_i_started_B(i,'M')
        ProcessList3.append(p)
        
    for process in ProcessList3:
        process.join()
        
    # log the seuccess ofjoins
    if ModVar.logFlag: My_Log.AllVisJoined('M')
    
    # calculate the finish time     
    RunTimVisB_Finish=time.perf_counter()
    dt4=RunTimVisB_Finish-RunTimVisB_Start
    dt4=round(dt1*100)/100
    
    # log elapsed time
    if ModVar.logFlag: My_Log.TimeVisFin(dt4,'M')
    
    # starting timer of Accuracy\Croo validation parallel processing
    RunTimAccB_Start=time.perf_counter()
    if ModVar.logFlag: My_Log.TimeAccStrt('M')
    
    # use threading
    ProcessList4=[]
    for i,C_svc in enumerate(ModVar.GAMMA_values):
        
        # define each thread
        p=threading.Thread(target=MLModel_instance.AccScor_B,args=(C_svc,ModVar.cvNum))
        
        # start the thread
        p.start()
        if ModVar.logFlag: My_Log.Train_CrossVal_i_started_PartB(i,'M')
        ProcessList4.append(p)
    
    # waiting for all Processes to join    
    for process in ProcessList4:
        process.join()
        
    # log the seuccess ofjoins
    if ModVar.logFlag: My_Log.AllAccJoined('M')
    
    # calculate the finish time     
    RunTimAccB_Finish=time.perf_counter()
    dt5=RunTimAccB_Finish-RunTimAccB_Start
    dt5=round(dt5*100)/100
    
    # total time
    dt6=dt4+dt5
    
    # log elapsed time
    if ModVar.logFlag: My_Log.TimeAccFin('M')
    
    # plotting train and validation errors
    ModFunc.PltTrnValErr_PartB()
    
    print('trnErrArray_B is:', ModVar.trnErrArray_B)
    print('valErrArray_B is :', ModVar.valErrArray_B)

    # it closes the log file
    if ModVar.logFlag: My_Log.filehandler.close() 
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------       
    # saving results
    if ModVar.saveplotFlag:
        ModFunc.SaveFigs()
        if ModVar.logFlag: My_Log.FigResSaved('M')
    
    # logging results of data analysis part A
    if ModVar.logFlag: My_Log.ResultsInfo_SentenceForm(dt1,dt2,dt3,dt4,dt5,dt6,'M')
    
    # logging results of data analysis part B
    if ModVar.logFlag: My_Log.ResultsInfo_TableForm(dt1,dt2,dt3,dt4,dt5,dt6,'M')
    #--------------------------------------------------------------------------




















