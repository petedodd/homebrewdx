# CODE EXAMPLES AND GRAPHS FOR: 'Simple inclusion of complex diagnostic algorithms 
#                                in infectious disease models for economic evaluation'
# 
# (C) PJ Dodd, JJ Pennington, L Bronner Murrison & DW Dowdy
# Code released under Creative Commons Attribution 4.0 International (CC BY 4.0) license
# http://creativecommons.org/licenses/by/4.0/
# You are free to use and adapt the code subject to this license
#
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.integrate import odeint
from scipy.optimize import fsolve
from copy import deepcopy as dcpy
from itertools import *
from SALib.sample import saltelli  # for simulation study
from SALib.analyze import sobol    # for simulation study
import pandas as pd             # for simulation study
import pickle
import seaborn as sns           # for simulation study plot

# set font types & styles for plotting
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
sns.set(style="whitegrid")


class Diagnostic:
    '''This is a class descibing a single diagnostic, for inputs (true disease states) that are
    I-, I+ and with outputs (test results) T-,T+.
    These can be chained together by specifying \'next\'
    and the getTables() method calculates the transition matrix for the subtree rooted at a given
    diagnostic, as well as the average costs and delayed for each transition. '''

    def __init__(self,sens,spec,cost,delay,ltfu,label=''):
        '''This constructs an instance of a diagnostic based on data
        on the sensitivity and specificity for I, the cost and delay incurred.
        '''
        self.sens = sens; self.spec = spec; self.cost = cost;self.delay = delay;self.ltfu = ltfu;
        self.label = label
        self.root = True        # by default, this is a root

        # todo - include ltfu
        # an array (I-,I+) x (T-,T+)
        self.transition = np.array([[   spec,    (1-spec) ],
                                    [ 1-sens,        sens ]])

        # 0=no tx, 1= tx
        self.next = [0,1]     # default as treatments, can be next diagnostics
        self.cost = cost
        self.delay = delay

    def __str__(self):
        '''For display purposes'''
        print self.transition
        return "Diagnostic instance %s, for specified patient characteristics. Data on characteristics and next test. Use help(instance) to learn more" % self.label

    def setnext(self,k,newdx):
        '''This is for setting the next diagnostic. Same as naively changing vector next, but takes care of root flag at the same time.'''
        if k not in [0,1]:
            print 'Error! k must be in 0..1'
        self.next[k] = dcpy(newdx)    # add to tree, as copy
        self.next[k].root = False      # record that this is no longer a root

    def getTables(self):
        '''To get the overall table of (I-,I+) x (no tx, tx), and similarly for costs and delays.
        Computes transitions as T_{ij} = \sum_k A_{ik}B^k_{ij}recursively.'''
        txo = np.zeros((2,2))   # transitions table
        cost = np.zeros((2,2))  # cost for in state getting to outcome etc.
        delay = np.zeros((2,2)) # delay
        if self.root:
            cost += self.cost * self.transition # add on own costs if root
            delay += self.delay * self.transition # add on own delay if root
        for k  in [0,1]:
            if isinstance(self.next[k],Diagnostic):
                txon, costn, delayn = self.next[k].getTables()
                for j in [0,1]:
                    nextbit = self.transition[:,k] * txon[:,j]
                    txo[:,j] += (1-self.ltfu) * nextbit
                    cost[:,j] += self.next[k].cost * (1-self.ltfu) * nextbit # new costs lookahead
                    # delay[:,j] += self.delay * (1-self.ltfu) * nextbit
                    delay[:,j] += self.next[k].delay * (1-self.ltfu) * nextbit #new try
            elif self.next[k] == k:
                txo[:,k] += (1-self.ltfu) * self.transition[:,k]
                cost[:,k] += 0
                delay[:,k] += 0           #new try
                # delay[:,k] += self.delay *  (1-self.ltfu) * self.transition[:,k]
            else:
                print "Invalid node in next for getTables! Need next[k]==k if not a subsequent test! label=%s, node=%j" % (self.label,j)
        return txo, cost, delay


class Algorithm(Diagnostic):
    ''' Defined from tree via inheritance'''
    def getFuns(self):
        '''This returns functions TPapp, TNapp, cost funs, QALY funs??
        '''
        # get tree data
        txo, cst, dly = self.getTables()   # calculate the relevant data
        Tz = dly.sum(axis=1)               #mean delay to outcome for TP/TN
        Cz = cst.sum(axis=1)               #mean cost to outcome for TP/TN

        def stinflowAp(X,t,i):
            ''' An approximation for TN (i=0), or TP (i=1)'''
            meanT = Tz[i]
            dX = np.zeros(3)
            dX[0] = inflow(t) - X[0]/meanT
            dX[1] = txo[i,0]*X[0]/meanT
            dX[2] = txo[i,1]*X[0]/meanT
            return dX

        # approximation: now states 1 and 2 correspond to -ve/+ve
        def TPap(X,t):
            return stinflowAp(X,t,1)

        def TNap(X,t):
            return stinflowAp(X,t,0)

        return TPap, TNap

#testing
# sens,spec,cost,delay,ltfu

s1 = .7
s2 = .7
sp1 = .9
sp2 = .9
T1 = 1.0
T2 = 1.0
C1 = 1.0
C2 = 10

test = Algorithm(s1,sp1,C1,T1,1e-9,label='testA')   #inherited
test2 = Algorithm(s2,sp2,C2,T2,1e-9,label='testB')  #inherited

print test
print test.getTables()


test3 = dcpy(test)

test3.setnext(1,test2)                    #test2 if +ve for test

print test3.getTables()


#  --------------- testing -------------
ta2 = np.array([[sp1+(1-sp1)*sp2, (1-sp1)*(1-sp2)],[1-s1+s1*(1-s2), s1*s2]])
print ta2

mt = test3.getTables()
print mt[0]

print T1 + s1*T2
print mt[2].sum(axis=1)
print mt[1].sum(axis=1)

print (1-s1+s1*(1-s2))
print mt[0][1,0]

print s1*s2
print mt[0][1,1]


siTPap, siTNap = test3.getFuns()          #new

eps = 0.0
def inflow(t):
    if eps >= 0:
        ans = 1.0-eps*t
    else:
        ans = -eps*t
    return ans

# cumulative
def cinflow(t):
    if eps >= 0:                          #decreasing
        ans = 1.0*t-eps*t*t/2
    else:
        ans = -eps*t*t
    return ans

# screening and confirmatory test (T1 then T2), state diagram:
#               ___ ___2=-ve
#             /      /
# Inflow -- 0=T1   /
#            \   /
#             1=T2
#                \_____3=+ve

# NB - these all have analytic solutions
# static model for true positives
def stinflowTP(X,t):
    ''' Looking at a 2-step diagnostic to -ve,+ve results
    0 and 1 are the two tests; 2 and 3 are -ve and +ve.
    For true positives.
    '''
    dX = np.zeros(4)
    dX[0] = inflow(t) - X[0]/T1
    dX[1] = s1*X[0]/T1 - X[1]/T2
    dX[2] = (1-s1)*X[0]/T1 + (1-s2)*X[1]/T2
    dX[3] = s2*X[1]/T2
    return dX


# approximation: now states 1 and 2 correspond to -ve/+ve
def stinflowTPap(X,t):
    ''' Appriximation
    For true positives.
    '''
    meanT = T1 + s1*T2
    dX = np.zeros(3)
    dX[0] = inflow(t) - X[0]/meanT
    dX[1] = (1-s1+s1*(1-s2))*X[0]/meanT
    dX[2] = s1*s2*X[0]/meanT
    return dX

# SSA approximation: now states 0 and 1 correspond to -ve/+ve
def stinflowTPssa(X,t):
    ''' Approximation
    SSA
    For true positives.
    '''
    dX = np.zeros(3)
    dX[0] = inflow(t) - X[0]/T1
    dX[1] = (1-s1)*X[0]/T1 + (1-s2)*(s1*X[0]/T1)
    dX[2] = s2*(s1*X[0]/T1)
    return dX


def stinflowTP(X,t):
    ''' Looking at a 2-step diagnostic to -ve,+ve results
    0 and 1 are the two tests; 2 and 3 are -ve and +ve.
    For true negatives.
    '''
    dX = np.zeros(4)
    dX[0] = inflow(t) - X[0]/T1
    dX[1] = s1*X[0]/T1 - X[1]/T2
    dX[2] = (1-s1)*X[0]/T1 + (1-s2)*X[1]/T2
    dX[3] = s2*X[1]/T2
    return dX



# static model for true negatives
def stinflowTN(X,t):
    ''' Looking at a 2-step diagnostic to -ve,+ve results
    0 and 1 are the two tests; 2 and 3 are -ve and +ve.
    For true negatives.
    '''
    dX = np.zeros(4)
    dX[0] = inflow(t) - X[0]/T1
    dX[1] = (1-sp1)*X[0]/T1 - X[1]/T2
    dX[2] = sp1*X[0]/T1 + sp2*X[1]/T2
    dX[3] = (1-sp2)*X[1]/T2
    return dX


# approximation static model for true negatives
def stinflowTNap(X,t):
    ''' Looking at a 2-step diagnostic to -ve,+ve results
    Approximation for true negatives.
    '''
    meanT = T1 + (1-sp1)*T2
    dX = np.zeros(3)
    dX[0] = inflow(t) - X[0]/meanT
    dX[1] = (sp1+(1-sp1)*sp2)*X[0]/meanT
    dX[2] = (1-sp1)*(1-sp2)*X[0]/meanT
    return dX



times = np.arange(0,30,.05)


# eps = -1.0/30

# ------ TPs-----------
# solve odes
Y0 = np.zeros(4)
TS = odeint(stinflowTP,Y0,times)

Y0ap = np.zeros(3)
TSap = odeint(stinflowTPap,Y0ap,times)

TSap2 = odeint(siTPap,Y0ap,times)

# proportion graph
fig = plt.figure()
line2, = plt.plot(times,TS[:,2]/(cinflow(times)+1e-6),label='Negative',color='k',linestyle='--')
line3, = plt.plot(times,TS[:,3]/(cinflow(times)+1e-6),label='Positive',color='k',linestyle='-')
line4, = plt.plot(times,TSap[:,1]/(cinflow(times)+1e-6),label='Negative (approximation)',color='red',linestyle='--')
line5, = plt.plot(times,TSap[:,2]/(cinflow(times)+1e-6),label='Positive (approximation)',color='red',linestyle='-')
plt.xlabel('Time')
plt.ylabel('Proportion')
plt.grid(True)
plt.legend(loc=4)
# plt.show()

# fn = 'graphs/propTPboth'+ str(eps) +'.pdf
fn = '../graphs/propTPbothC.pdf'
fig.savefig(fn,format='PDF')


# proportion graph (same as above but B&W)
fig = plt.figure()
line2, = plt.plot(times,TS[:,2]/(cinflow(times)+1e-6),label='Negative',color='k',linestyle='--')
line3, = plt.plot(times,TS[:,3]/(cinflow(times)+1e-6),label='Positive',color='k',linestyle='-')
line4, = plt.plot(times,TSap[:,1]/(cinflow(times)+1e-6),label='Negative (approximation)',color='tab:gray',linestyle='--')
line5, = plt.plot(times,TSap[:,2]/(cinflow(times)+1e-6),label='Positive (approximation)',color='tab:gray',linestyle='-')
plt.xlabel('Time')
plt.ylabel('Proportion')
plt.grid(True)
plt.legend(loc=4)
fn = '../graphs/propTPboth.pdf'
fig.savefig(fn,format='PDF')


# ------------------------ SIR model -----------------
s1 = .7
s2 = .7
sp1 = .9
sp2 = .9
T1 = 0.1
T2 = 0.1
C1 = 1.0
C2 = 10
R0 = 2
tnr = 1e-1                                #true negative rate of dx
tpr = 1e-0                                 #true positive rate of dx
Trx = 1                                   #treatment duration

crosslines = 1                            #is there infection and progression during diagnosis?

times = np.arange(0,30,.05)
X0 = np.array([1e5-10,10,0])


def SIR(X,t):
    ''' This defines the SIR equations'''
    # helper quantities
    N = np.sum(X)                         #total population
    endtx = np.array([[0,0,0],[0,-1,0],[0,1,0]])       #matrix for ending tx
    etxi = np.array([0,1,0])       #vector for ending tx for I
    ontx = np.array([0,1,0])                  #on tx
    notx = np.array([1,0,0])                 #no previous tx
    infectious = np.array([1,0,1])           #mask for infectiousness

    # state variables
    S = X[0:3]                            #[no treatment history, on tx, treated]
    I = X[3:6]
    R = X[6:9]
    SD1, SD2 = X[9], X[10]                 #dx1 and dx2 for S
    ID1, ID2 = X[11], X[12]                 #dx1 and dx2 for I
    RD1, RD2 = X[13], X[14]               #dx1 and dx2 for I

    # equations
    foi = ( sum(I*infectious) + ID1 + ID2  )/N             #force-of-infection
    dS = -R0*S*foi  - notx*tnr*S + notx*sp1*SD1/T1 + (ontx*(1-sp2)+notx*sp2)*SD2/T2  + np.dot(endtx,S/Trx)
    dI = R0*S*foi - I   - notx*(tnr+tpr)*I + notx*(1-s1)*ID1/T1 + (ontx*s2 + notx*(1-s2))*ID2/T2 - np.dot(etxi,I/Trx)*etxi
    dR = I  - notx*tnr*R + notx*sp1*RD1/T1 + (ontx*(1-sp2)+notx*sp2)*RD2/T2 + np.dot(endtx,R/Trx) + np.dot(etxi,I/Trx)*np.array([0,0,1])
    dSD1 = tnr * S[0] - SD1/T1  - foi*SD1*crosslines
    dSD2 = (1-sp1)*SD1/T1 - SD2/T2  - foi*SD2*crosslines
    dID1 = (tnr+tpr)*I[0] - ID1/T1  + (foi*SD1 - ID1)*crosslines
    dID2 = s1*ID1/T1 - ID2/T2 + (foi*SD2  - ID2)*crosslines
    dRD1 = tnr*R[0] - RD1/T1  + ID1*crosslines
    dRD2 = (1-sp1)*RD1/T1 - RD2/T2  + ID2*crosslines

    # results
    dX = np.array( list(dS) + list(dI) + list(dR) + [dSD1,dSD2,dID1,dID2,dRD1,dRD2] )
    return dX

# test
# x0 = np.ones(15)

x0 = np.array([1e5-10,0,0, 10,0,0, 0,0,0] + [0]*3*2)

print SIR(np.ones(15),0)

print SIR(np.ones(15),0).sum()

print SIR(x0,0).sum()

# prb = 1.0/(1+tpr+tnr)
# Rcor = (1+(1-prb)*(T1+s1*T2)) / (1-(1-prb)*(1-s1))
# print Rcor

# #
# crosslines = 1                            #is there infection and progression during diagnosis?
# T1=.1
# T2=.1
# R0 = 5                                    #take recovery rate =1
# tnr = 1e-1                                #true negative rate of dx
# tpr = 1e-0                                 #true positive rate of dx
# Trx = 1                                   #treatment duration

TS = odeint(SIR,x0,times)


# developing an approximation via the Diagnostic class
class Algorithm(Diagnostic):
    ''' Defined from tree via inheritance'''
    def getFuns(self):
        '''This returns functions TPapp, TNapp, cost funs, QALY funs??
        '''
        # get tree data
        txo, cst, dly = self.getTables()   # calculate the relevant data
        Tz = dly.sum(axis=1)               #mean delay to outcome for TP/TN
        Cz = cst.sum(axis=1)               #mean cost to outcome for TP/TN

        # necessary functions
        def flows(dxpop,foi):
            ''' An approximation for the flows back and on/through for this diagnostic.
            i measures true positive or not. For use in ODEs.
            '''
            Svec = np.array( list( dxpop[0] * txo[0,:] / Tz[0] ) + [0] )#for S - TN
            Ivec = np.array( list( dxpop[1] * txo[1,:] / Tz[1] ) + [0] )#for I - TP
            Rvec = np.array( list( dxpop[2] * txo[0,:] / Tz[0] ) + [0] )#for R - TN
            ans = (Svec, Ivec, Rvec)                      #initialize tuple
            return ans #tuple of 3-arrays convenient form for ODEs

        # approximation to cost
        def cost(M):                      #takes the count in SIR, notx as timeseries matrix
            R = np.repeat( np.array([tnr,tnr+tpr,tnr]), M.shape[0] ).reshape((M.shape[0],3),order='F')
            MR = M*R                      #rates of entering dx
            return (MR[:,0] + MR[:,2]) * Cz[0] + MR[:,1] * Cz[1]   # returns timeseries

        return flows, Tz, cost





def SIRap(X,t):
    ''' This defines the approximate SIR equations'''
    # helper quantities
    N = np.sum(X)                         #total population
    endtx = np.array([[0,0,0],[0,-1,0],[0,1,0]])       #matrix for ending tx
    etxi = np.array([0,1,0])       #vector for ending tx for I
    notx = np.array([1,0,0])                 #no previous tx
    infectious = np.array([1,0,1])           #mask for infectiousness

    # state variables
    S = X[0:3]                            #[no treatment history, on tx, treated]
    I = X[3:6]
    R = X[6:9]
    dx = X[9:12]                          #'in diagnosis'

    # equations
    foi = ( sum(I*infectious) + dx[1] )/N             #force-of-infection
    dxflow = np.array([ tnr*S[0], (tnr+tpr)*I[0], tnr*R[0] ])   #rates of starting diagnosis
    flowS, flowI, flowR = tflows(dx,foi) #get out-flows from diagnostics

    # ODEs
    dS = -R0*S*foi  - notx*dxflow[0]   + np.dot(endtx,S/Trx)  + flowS
    dI = R0*S*foi - I   - notx*dxflow[1]  - np.dot(etxi,I/Trx)*etxi  + flowI
    dR = I  - notx*dxflow[2]  + np.dot(endtx,R/Trx) + np.dot(etxi,I/Trx)*np.array([0,0,1])  + flowR
    if crosslines > 0:
        dxflow += np.array([0,dx[0] * foi, dx[1] * 1])   #transitions to other dx states
    ddx = dxflow - dx / np.array([Tz[0]/(1+crosslines*foi*Tz[0]), Tz[1]/(1+crosslines*1*Tz[1]), Tz[0]])   #in minus out (relevant timescales: extras for progression )

    # results
    dX = np.array( list(dS) + list(dI) + list(dR) + list(ddx) )
    return dX


# testing ----------
# SIR = 036
# SIR on tx = 147
# SIR > tx = 258

x0 = np.array([1e5-10,0,0, 10,0,0, 0,0,0] + [0]*3*2)
times = np.arange(0,30,.02)


# ------------------------------------------------ define algorithm
# T1=.1*1e0                                     #dx1 timescale
# T2=.1*1e0                                     #dx2 timescale
# R0 =  2                                    #take recovery rate =1
# tnr = 1e-1                                #true negative rate of dx
# tpr = 1e-0                                 #true positive rate of dx
# Trx = 1                                   #treatment duration

# crosslines = 1*1          #is there infection and progression during diagnosis?
# C1 = 1.0                                  #cost test 1
# C2 = 1                                    #cost test 2

# s1=.7**1
# sp1 = .9**1
# s2=.7**1
# sp2 = .9**1


test = Algorithm(s1,sp1,C1,T1,1e-9,label='testA')   #inherited
test2 = Algorithm(s2,sp2,C2,T2,1e-9,label='testB')  #inherited
test3 = dcpy(test)
test3.setnext(1,test2)                    #test2 if +ve for test


# A,B,C = test3.getTables()

tflows, Tz, cost = test3.getFuns()                  #get functions for differential equations


# solve
TS = odeint(SIR,x0,times)
TSap = odeint(SIRap,x0[0:12],times)

# the SIR bits
plt.close('all')
fig = plt.figure()
plt.plot(times,TS[:,0],label='S, no tx',color='k',linestyle='-')
plt.plot(times,TS[:,3],label='I, no tx',color='k',linestyle='--')
plt.plot(times,TS[:,6],label='R, no tx',color='k',linestyle='-.')
plt.plot(times,TSap[:,0],label='S, no tx (approximation)',color='tab:gray',linestyle='-')
plt.plot(times,TSap[:,3],label='I, no tx (approximation)',color='tab:gray',linestyle='--')
plt.plot(times,TSap[:,6],label='R, no tx (approximation)',color='tab:gray',linestyle='-.')
plt.grid(True)
plt.legend(loc=1,prop={'size':10})
plt.xlabel('Time')
plt.ylabel('Number')
# plt.show()

fig.savefig('../graphs/SITR'+str(crosslines)+'.pdf',format='PDF')
 

plt.close('all')
fig = plt.figure()
plt.plot(times,TS[:,6:9].sum(axis=1) + TS[:,13:15].sum(axis=1),label='Infections',color='k',linestyle='-')
plt.plot(times,TSap[:,6:9].sum(axis=1) + TSap[:,11],label='Infections (approximation)',color='tab:gray',linestyle='-')
plt.plot(times,TS[:,2]+TS[:,5]+TS[:,8],label='Treatments',color='k',linestyle='--')
plt.plot(times,TSap[:,2]+TSap[:,5]+TSap[:,8],label='Treatments (approximation)',color='tab:gray',linestyle='--')
plt.ylabel('Cumulative number')
plt.xlabel('Time')
plt.legend(loc=0,prop={'size':10})
plt.grid(True)
fig.savefig('../graphs/SITRcumulative'+str(crosslines)+'a.pdf',format='PDF')

plt.close('all')
fig = plt.figure()
cts = np.cumsum((C1*TS[:,np.array([9,11,13])]/T1).sum(axis=1))+np.cumsum((C2*TS[:,np.array([10,12,14])]/T2).sum(axis=1)) #cost time series
ctsap = np.cumsum( cost( TSap[:,np.array([0,3,6])] ) )
plt.plot(times,cts,label='Diagnostics',color='k',linestyle='-')
plt.plot(times,ctsap,label='Diagnostics (approximation)',color='tab:gray',linestyle='-')
plt.xlabel('Time')
plt.ylabel('Cumulative cost')
plt.legend(loc=0,prop={'size':10})
plt.grid(True)
fig.savefig('../graphs/SITRcumulative'+str(crosslines)+'b.pdf',format='PDF')


# # cumulative comparison in one plot
# plt.close('all')
# fig = plt.figure(figsize=(10,5))
# plt.subplot(121)
# plt.plot(times,TS[:,6:9].sum(axis=1) + TS[:,13:15].sum(axis=1),label='infections',color='k',linestyle='-')
# plt.plot(times,TSap[:,6:9].sum(axis=1) + TSap[:,11],label='infections (approximation)',color='red',linestyle='-')
# plt.plot(times,TS[:,2]+TS[:,5]+TS[:,8],label='treatments',color='k',linestyle='--')
# plt.plot(times,TSap[:,2]+TSap[:,5]+TSap[:,8],label='treatments (approximation)',color='red',linestyle='--')
# plt.ylabel('cumulative number')
# plt.xlabel('time')
# plt.legend(loc=0,prop={'size':10})
# plt.grid(True)
# plt.subplot(122)
# cts = np.cumsum((C1*TS[:,np.array([9,11,13])]/T1).sum(axis=1))+np.cumsum((C2*TS[:,np.array([10,12,14])]/T2).sum(axis=1)) #cost time series
# ctsap = np.cumsum( cost( TSap[:,np.array([0,3,6])] ) )
# plt.plot(times,cts,label='diagnostics',color='k',linestyle='-')
# plt.plot(times,ctsap,label='diagnostics (approximation)',color='red',linestyle='-')
# plt.xlabel('time')
# plt.ylabel('cumulative cost')
# plt.legend(loc=0,prop={'size':10})
# plt.grid(True)
# plt.tight_layout()
# # plt.show()
# fig.savefig('graphs/SITRcumulative'+str(crosslines)+'.pdf',format='PDF')

# ------------------- comparing errors --------------------------

# SIR = 036
# SIR on tx = 147
# SIR > tx = 258

# print test3.getTables()[2]

# # cumulative treatments
# txtrue = TS[:,2]+TS[:,5]+TS[:,8]
# txappn = TSap[:,2]+TSap[:,5]+TSap[:,8]
# txerr = (txtrue-txappn)/(txtrue + 1e-10)



# # R post treatment
# rerr = (TS[:,6] -TSap[:,6])/(TS[:,6] + 1e-10)   #recovered no tx

# print txerr[len(txerr)-1]                 #-16%
# print rerr[len(rerr)-1]                   #+6%

# check about infection during dx process...
# looking at error as a function of timescale
times2 = np.arange(0,30,.01)
nn = 500                                #number of observations
TF = np.linspace(2e-3,1,nn)             #fractional times
PEI = np.zeros(nn)                      #proportional error infections
PEC = np.zeros(nn)                      #proportional error cost
PET = np.zeros(nn)                      #proportional error treatments

# generate data
for i in range(len(TF)):
    if (i%100==0): print i
    T1 = T2 = TF[i]
    test = Algorithm(s1,sp1,C1,T1,1e-9,label='testA')   #inherited
    test2 = Algorithm(s2,sp2,C2,T2,1e-9,label='testB')  #inherited
    test3 = dcpy(test)
    test3.setnext(1,test2)                    #test2 if +ve for test
    tflows, Tz, cost = test3.getFuns()                  #get functions for differential equations
    TS = odeint(SIR,x0,times)
    TSap = odeint(SIRap,x0[0:12],times)
    cts = np.cumsum((C1*TS[:,np.array([9,11,13])]/T1).sum(axis=1))+np.cumsum((C2*TS[:,np.array([10,12,14])]/T2).sum(axis=1)) #cost time series
    ctsap = np.cumsum( cost( TSap[:,np.array([0,3,6])] ) )
    nl = TS.shape[0] - 1
    PEI[i] = (TSap[nl,6:9].sum() + TSap[nl,11])/(TS[nl,6:9].sum() + TS[nl,13:15].sum()) - 1
    PET[i] = (TSap[nl,2]+TSap[nl,5]+TSap[nl,8])/(TS[nl,2]+TS[nl,5]+TS[nl,8]) - 1
    PEC[i] = (ctsap[nl])/(cts[nl]) - 1


xz = TF
xlbl = 'Diagnostic timescale over infection timescale'

# the SIR bits
plt.close('all')
fig = plt.figure()
plt.plot(xz,(PEI),label='Infections',color='k',linestyle='-')
plt.plot(xz,(PET),label='Treatments',color='k',linestyle='--')
plt.plot(xz,(PEC),label='Diagnostic cost',color='k',linestyle='-.')
plt.grid(True)
plt.ylabel('Proportional error in cumulative total')
plt.xlabel(xlbl)   #B
plt.legend(loc=0)# ,prop={'size':10})
# plt.show()
fig.savefig('../graphs/SITRerr'+str(crosslines)+'.pdf',format='PDF')


# ======== More general simulation study across parameters =======

pmz = ['T1','T2','s1','s2','sp1','sp2','C1','C2','R0','xinit','tnr','tpr','Trx']
bds = [ [.01,1.0],[.01,1.0], [.5,1.],[.5,1.],[.5,1.],[.5,1.], [.5,1.5],[.5,1.5], [1.5,3], [1.,10.], [0.01,1.],[0.01,1.],[.5,1.5]]
print dict(zip(pmz,bds))


problem = {
    'num_vars': len(pmz),
    'names': pmz,
    'bounds': bds
}

# Generate samples
param_values = saltelli.sample(problem, 1000, calc_second_order=False)
print param_values.shape

# Run model (example)
nn = param_values.shape[0]
Y = np.zeros(nn)                      #proportional error infections
for i in range(nn):
    if (i%100==0): print i
    T1,T2,s1,s2,sp1,sp2,C1,C2,R0,xinit,tnr,tpr,Trx = tuple(param_values[i,:])
    tpr = 1.0/(0.5 + tpr)
    tnr = 0.1/(0.5 + tnr)
    x1 = x0.copy()
    x1[3] = x1[3] * xinit                     # rescale
    test = Algorithm(s1,sp1,C1,T1,1e-9,label='testA')   #inherited
    test2 = Algorithm(s2,sp2,C2,T2,1e-9,label='testB')  #inherited
    test3 = dcpy(test)
    test3.setnext(1,test2)                    #test2 if +ve for test
    tflows, Tz, cost = test3.getFuns() #get functions for differential equations
    TS = odeint(SIR,x0,times)
    TSap = odeint(SIRap,x0[0:12],times)
    # treatments
    tmpn = (TSap[:,2] + TSap[:,5] + TSap[:,8])
    tmpd = (TS[:,2] + TS[:,5] + TS[:,8] + 1e-6)
    Y[i] = max(tmpn/tmpd) - 1


# Perform analysis
Si = sobol.analyze(problem, Y,calc_second_order=False,print_to_console=True)

# Print the first-order sensitivity indices
print dict(zip(pmz,Si['S1']))

# latex versions
pmz[0] = r'$T_1$'
pmz[1] = r'$T_2$'
pmz[2] = r'$s_1$'
pmz[3] = r'$s_2$'
pmz[4] = r'$sp_1$'
pmz[5] = r'$sp_2$'
pmz[6] = r'$C_1$'
pmz[7] = r'$C_2$'
pmz[8] = r'$\beta$'
pmz[9] = r'$x_0$'
pmz[10] = r'$a^{-1}$'
pmz[11] = r'$b^{-1}$'
pmz[12] = r'$T_T$'


# # # save out
# import pickle
# pickle.dump( param_values, open( "params.pkl", "wb" ) )
# pickle.dump( Y, open( "Y.pkl", "wb" ) )
# param_values = pickle.load( open( "params.pkl", "rb" ) )
# Y = pickle.load( open( "Y.pkl", "rb" ) )

# make df from them
dt = {'Variable':pmz,'Sobol total effect':Si['ST']}
df = pd.DataFrame.from_dict(dt)
df.sort(columns='Sobol total effect',inplace=True,ascending=False)

# plotting
plt.close('all')
f = plt.figure(figsize=(5, 5))
sns.barplot(x='Sobol total effect',y='Variable',data=df,color='black')
fn = '../graphs/Sobol.pdf'
# f.savefig(fn)
f.savefig(fn,format='PDF')

