
#==============================================================================
# Created on Feb 10, 2020
#
# @author: Ramin Mehdizad Tekiyeh

# This code is written in Python 3.7.4 , Spyder 3.3.6
#==============================================================================


#==============================================================================
# This module contains all the functions that are used in the main program
#==============================================================================


#==============================================================================
# importing standard classes
#==============================================================================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import os.path


#==============================================================================
# importing module codes
#==============================================================================
import ModVar


#==============================================================================
# this function asks user for input data
#==============================================================================  
def Input():
    
    print('')
    print('|===========================================================')
    print('|  ==> To run the code with default values, just press Enter')
    print('|  ==> Otherwise:')
    print('|  ==> Enter the parameters as following format:')
    print('|')
    print('|       -d D:/p/DataSet1.csv -w D:/p -p 0 -l 0')
    print('|')
    print('|  ==> To get help, type "-h" and press Enter')
    print('|  ==> To exit program, type "Q" and press Enter')
    print('|===========================================================')
    
    ModVar.str1=input('  Enter parameters: ').strip()


#==============================================================================
# this function creates and sets result directory
#==============================================================================   
def CreateAndSetResultsDir():
    global path
    
    # the work directory is set in arg-parse
    # get current path
    path=ModVar.args.WorkDir 
    
    # creating results directory and avoid to oveeride the previous results
    i=1
    ResFldExst=True
    while ResFldExst==True:
        resultspath = path+"\\Results_Run_"+str(i)
        ResFldExst=os.path.exists(resultspath)
        i+=1
    ModVar.resultssubpath=resultspath
    # no we create results sub folder and set it as result path
    os.mkdir(ModVar.resultssubpath)
    os.chdir(ModVar.resultssubpath)
    
 
#==============================================================================
# this function inputs the parameters for Part A of the problem 
#==============================================================================
def CreateSubPlotStructure_PartA(x):
    global axes,cmap,xMesh,yMesh,fig_SubPlotC
    
    nrows=len(ModVar.C_values)//3  if len(ModVar.C_values)%3==0 else len(ModVar.C_values)//3+1
    fig_SubPlotC,axes=plt.subplots(nrows=nrows,ncols=3,figsize=(15,5.0*nrows))
    cmap=ListedColormap(['#b30065','#178000'])
    
    xMin,xMax=x[:,0].min()-1,x[:,0].max()+1
    yMin,yMax=x[:,1].min()-1,x[:,1].max()+1
    xMesh,yMesh=np.meshgrid(np.arange(xMin,xMax,0.01),np.arange(yMin,yMax,0.01))
    
 
#==============================================================================
# this function inputs the parameters for Part B of the problem 
#==============================================================================
def CreateSubPlotStructure_PartB(x):
    global axes,cmap,xMesh,yMesh,fig_SubPlotGamma
    
    nrows=len(ModVar.GAMMA_values)//3  if len(ModVar.GAMMA_values)%3==0 else len(ModVar.GAMMA_values)//3+1
    fig_SubPlotGamma,axes=plt.subplots(nrows=nrows,ncols=3,figsize=(15,5.0*nrows))
    cmap=ListedColormap(['#b30065','#178000'])
    
    xMin,xMax=x[:,0].min()-1,x[:,0].max()+1
    yMin,yMax=x[:,1].min()-1,x[:,1].max()+1
    xMesh,yMesh=np.meshgrid(np.arange(xMin,xMax,0.01),np.arange(yMin,yMax,0.01))


#==============================================================================
# this function gets computer spec
#==============================================================================
def GetSysInfo():
    import platform,socket,re,uuid,psutil
    try:
        ModVar.sysinfo.append(['platform',platform.system()]) 
        ModVar.sysinfo.append(['platform-release',platform.release()])
        ModVar.sysinfo.append(['platform-version',platform.version()])
        ModVar.sysinfo.append(['architecture',platform.machine()])
        ModVar.sysinfo.append(['hostname',socket.gethostname()])
        ModVar.sysinfo.append(['ip-address',socket.gethostbyname(socket.gethostname())])
        ModVar.sysinfo.append(['mac-address',':'.join(re.findall('..', '%012x' % uuid.getnode()))])
        ModVar.sysinfo.append(['processor',platform.processor()])
        ModVar.sysinfo.append(['ram',str(round(psutil.virtual_memory().total / (1024.0 **3)))+" GB"])
    
    except Exception as e:
        print(e)
        

#==============================================================================
# this function sets the parameters of problem
#==============================================================================
def ProgParameters():
    global nsamples, m_noisy,GAMMA_PartA, C_values,trnErr, valErr, tFrac, vFrac
    global kernel,random_state_SVC,cvNum 

    ModVar.tFrac=0.2
    ModVar.vFrac=0.2
    ModVar.GAMMA_PartA='scale'
    ModVar.C_PartB=10
    ModVar.kernel='rbf'
    ModVar.random_state_SVC=0
    ModVar.cvNum=10
    ModVar.C_values=np.power(10.0,np.arange(-3.0, 6.0, 1.0))
    print(ModVar.C_values)
    ModVar.GAMMA_values=np.power(10.0,np.arange(-2, 4, 1.0))
    print(ModVar.GAMMA_values)
    
    ModVar.figdpi=200

 
#==============================================================================
# this function plots train and validation errors of Part A
#==============================================================================
def PltTrnValErr_PartA():
    global fig_TrnVal_PartA
    
    # initializethe lists
    trnErrList_A=[]
    valErrList_A=[]
    
    # extracting data from Dict to List
    fig_TrnVal_PartA=plt.figure(80)
    for i,(xx,yy) in enumerate(ModVar.trnErr_A.items()):
        trnErrList_A.append([xx,yy])
        
    for i,(xx,yy) in enumerate(ModVar.valErr_A.items()):
        valErrList_A.append([xx,yy])
        
    # we need to sort because they are added to the list as soon as they finish
    # using sort() + lambda to sort list of list (sort by second index) 
    trnErrList_A.sort(key=lambda trnErrList_A:trnErrList_A[0]) 
    valErrList_A.sort(key=lambda valErrList_A:valErrList_A[0]) 
    
    ModVar.trnErrArray_A=np.array(trnErrList_A)
    ModVar.valErrArray_A=np.array(valErrList_A)

    #plotting train validation plot
    plt.plot(np.log10(ModVar.trnErrArray_A[:,0]),ModVar.trnErrArray_A[:,1],'*',color='black')
    plt.plot(np.log10(ModVar.valErrArray_A[:,0]),ModVar.valErrArray_A[:,1],'*',color='black') 
    
    plt.plot(np.log10(ModVar.trnErrArray_A[:,0]),ModVar.trnErrArray_A[:,1],'-',color='blue', label='train') 
    plt.plot(np.log10(ModVar.valErrArray_A[:,0]),ModVar.valErrArray_A[:,1],'-',color='red', label='validation')
    
    plt.legend(loc='lower right',frameon=False)
    plt.xlabel('log(C values)') 
    plt.ylabel('Accuracy') 
    
    
#==============================================================================
# this function plots train and validation errors of Part B
#==============================================================================
def PltTrnValErr_PartB():
    global fig_TrnVal_PartB
    
    # initializethe lists
    trnErrList_B=[]
    valErrList_B=[]
    
    # extracting data from Dict to List
    fig_TrnVal_PartB=plt.figure(50)
    for i,(xx,yy) in enumerate(ModVar.trnErr_B.items()):
        trnErrList_B.append([xx,yy])
        
    for i,(xx,yy) in enumerate(ModVar.valErr_B.items()):
        valErrList_B.append([xx,yy])
        
    # we need to sort because they are added to the list as soon as they finish
    # using sort() + lambda to sort list of list (sort by second index) 
    trnErrList_B.sort(key=lambda trnErrList_B:trnErrList_B[0]) 
    valErrList_B.sort(key=lambda valErrList_B:valErrList_B[0]) 
    
    ModVar.trnErrArray_B=np.array(trnErrList_B)
    ModVar.valErrArray_B=np.array(valErrList_B)

    #plotting train validation plot
    plt.plot(np.log10(ModVar.trnErrArray_B[:,0]),ModVar.trnErrArray_B[:,1],'*',color='black')
    plt.plot(np.log10(ModVar.valErrArray_B[:,0]),ModVar.valErrArray_B[:,1],'*',color='black') 
    
    plt.plot(np.log10(ModVar.trnErrArray_B[:,0]),ModVar.trnErrArray_B[:,1],'-',color='blue', label='train') 
    plt.plot(np.log10(ModVar.valErrArray_B[:,0]),ModVar.valErrArray_B[:,1],'-',color='red', label='validation')
    
    plt.legend(loc='lower right',frameon=False)
    plt.xlabel('log(Gamma values)') 
    plt.ylabel('Accuracy')     
    
    
#==============================================================================
# this function inputs the parameters of the problem
#==============================================================================
def PlotRawData(x_data,y_data):    
    # Plot csv data and save it to a jpg file
    global fig_XY
    fig_XY=plt.figure()
    cmap=ListedColormap(['#b30065','#178000'])
    plt.scatter(x_data[:,0],x_data[:,1],c=y_data,cmap=cmap,edgecolors='k')
    plt.xlabel('x1') 
    plt.ylabel('x2')    


#==============================================================================
# this function prinrts parsed data
#==============================================================================
def PrintParsedData(): 
    print('') 
    print('  =====================Parsed  Data==================')  
    print('  ', ModVar.args)
    print('')
    print('  Script Path          =', ModVar.args.ScriptPath)
    print('  .csv file address    =', ModVar.args.DataBasePath)
    print('  Work Directory       =', ModVar.args.WorkDir)
    # results path is ceated ans set in CreateAndSetResultsDir()
    print('  Results Path         =', os.getcwd())
    print('  Save Plots           =', ModVar.args.SavePlot)
    print('  Create Log File      =', ModVar.args.logFile)
    print('  ===================================================')
    print('')
    

#==============================================================================
# this function sets flags for log and save plot
#==============================================================================        
def SetFlags():        

    if ModVar.args.SavePlot=='0':
        ModVar.saveplotFlag=False
    else:
        ModVar.saveplotFlag=True

    if ModVar.args.logFile=='0':
        ModVar.logFlag=False
    else:
        ModVar.logFlag=True

       
#==============================================================================
# this function saves figures and results
#==============================================================================        
def SaveFigs():        
    # saving figures of Part A
    dpii=ModVar.figdpi
    fig_XY.savefig('DataScatter.jpg',dpi=dpii)
    fig_SubPlotC.savefig('C_Effect.jpg',dpi=dpii)
    fig_TrnVal_PartA.savefig('Trn_Val_C.jpg',dpi=dpii)
    # saving figures of Part B
    fig_SubPlotGamma.savefig('Gamma_Effect.jpg',dpi=dpii)
    fig_TrnVal_PartB.savefig('TrnValGamma.jpg',dpi=dpii)

                
#==============================================================================
# this function does the calculation of prediction and prepare plot data Part A
#==============================================================================
def visualize_PartA(p,clf,param,x,y,index):
    
    # train the model
    clf_trained=clf.fit(x,y)
    
    # we save the trained models to be used later in cross val
    ModVar.SVCTrainedModelsDict_C[p]=clf_trained
    
    # define which plot to be used for this svc model
    r,c=np.divmod(index,3)
    ax=axes[r,c]
    print("index,r,c",index,r,c)

    # Plot contours
    zMesh=clf.decision_function(np.c_[xMesh.ravel(),yMesh.ravel()])
    zMesh=zMesh.reshape(xMesh.shape)
    ax.contourf(xMesh,yMesh,zMesh,cmap=plt.cm.PiYG,alpha=0.6)
    
    ax.contour(xMesh,yMesh,zMesh,colors='k',levels=[-1,0,1],
               alpha=0.5,linestyles=['--','-','--'])

    # Plot data
    ax.scatter(x[:,0],x[:,1],c=y,cmap=cmap,edgecolors='k')
    ax.set_title('{0}={1}'.format(param, p))
    
    
#==============================================================================
# this function does the calculation of prediction and prepare plot data Part B
#==============================================================================
def visualize_PartB(p,clf,param,x,y,index):
    
    # train the model
    clf_trained=clf.fit(x,y)
    
    # we save the trained models to be used later in cross val
    ModVar.SVCTrainedModelsDict_Gamma[p]=clf_trained
    
    # define which plot to be used for this svc model
    r,c=np.divmod(index,3)
    ax=axes[r,c]
    print("index,r,c",index,r,c)

    # Plot contours
    zMesh=clf.decision_function(np.c_[xMesh.ravel(),yMesh.ravel()])
    zMesh=zMesh.reshape(xMesh.shape)
    ax.contourf(xMesh,yMesh,zMesh,cmap=plt.cm.PiYG,alpha=0.6)
    
    ax.contour(xMesh,yMesh,zMesh,colors='k',levels=[-1,0,1],
               alpha=0.5,linestyles=['--','-','--'])

    # Plot data
    ax.scatter(x[:,0],x[:,1],c=y,cmap=cmap,edgecolors='k')
    ax.set_title('{0}={1}'.format(param, p))
    
   

        


 
    

















