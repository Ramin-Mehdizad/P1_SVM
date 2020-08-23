
#==============================================================================
# Created on Feb 10, 2020
#
# @author: Ramin Mehdizad Tekiyeh

# This code is written in Python 3.7.4 , Spyder 3.3.6
#==============================================================================


#==============================================================================
# This module holds the variables that are shared between modules
#==============================================================================


#==============================================================================
# initializing parameters of single number variables
#==============================================================================
nsamples=0
n_samples=0
m_noisy=0
m_noisy_Frac=0
GAMMA_PartA=0
C_PartB=10
C_values=0
GAMMA_values=0
tFrac=0
vFrac=0
kernel=0
random_state_SVC=0
cvNum=0

# figure quality
figdpi=200

str1=''
args=''


#==============================================================================
# initializing Flags
#==============================================================================
logFlag=False
saveplotFlag=False


#==============================================================================
# initializing sharing folder path
#==============================================================================
resultssubpath=''


#==============================================================================
# initializing Dict variables
#==============================================================================
trnErr_A=dict()
valErr_A=dict()
trnErr_B=dict()
valErr_B=dict()
SVCTrainedModelsDict_C=dict()
SVCTrainedModelsDict_Gamma=dict()


#==============================================================================
# initializing Lists
#==============================================================================
trnErrArray_A=[]
valErrArray_A=[]
trnErrArray_B=[]
valErrArray_B=[]
x_data=[]
y_data=[]
sysinfo=[]











