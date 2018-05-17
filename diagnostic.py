# CODE EXAMPLE FOR: 'Simple inclusion of complex diagnostic algorithms 
#                                in infectious disease models for economic evaluation'
# 
# (C) PJ Dodd, JJ Pennington, L Bronner Murrison & DW Dowdy
# Code released under Creative Commons Attribution 4.0 International (CC BY 4.0) license
# http://creativecommons.org/licenses/by/4.0/
# You are free to use and adapt the code subject to this license
#
# This file defines a base diagnostic class and imports some libraries (2 next lines)
import numpy as np                        #for arrays
from copy import deepcopy as dcpy         #for copying classes

class Diagnostic:
    def __init__(self,sens,spec,cost,delay,ltfu):   #initialize
        self.sens = sens; self.spec = spec; 
        self.cost = cost; self.delay = delay;
        self.ltfu = ltfu;     # loss to follow-up
        self.root = True      # by default, this is a root
        self.next = [0,1]     # default as treatments, can be next diagnostics
        self.transition = np.array([[spec,(1-spec) ],
                                    [1-sens,sens ]])

    def setnext(self,k,newdx):       # next test
        self.next[k] = dcpy(newdx)    # add to tree, as copy
        self.next[k].root = False     # record that this is no longer a root

    def getTables(self):                  # get matrices by recursion
        txo = np.zeros((2,2)); cost = np.zeros((2,2)); delay = np.zeros((2,2))
        if self.root:
            cost += self.cost*self.transition # add on own costs if root
        for k  in [0,1]:
            if isinstance(self.next[k],Diagnostic):
                txon, costn, delayn = self.next[k].getTables()
                for j in [0,1]:
                    nextbit = self.transition[:,k] * txon[:,j]
                    txo[:,j] += (1-self.ltfu) * nextbit
                    cost[:,j] += self.next[k].cost * (1-self.ltfu) * nextbit 
                    delay[:,j] += self.delay * (1-self.ltfu) * nextbit
            elif self.next[k] == k:
                txo[:,k] += (1-self.ltfu) * self.transition[:,k]
                cost[:,k] += 0            
                delay[:,k] += self.delay * (1-self.ltfu) * self.transition[:,k]
        return txo, cost, delay
