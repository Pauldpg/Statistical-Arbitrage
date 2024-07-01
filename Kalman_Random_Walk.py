# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 22:07:35 2018

@author: Guillaume
"""

import numpy as np
import math
import scipy as sp


def Kalman_RW(y,z,sr):
    global _Q, _H, _Y, _Z, _X0, _P0, _A, _nZ, _sr
    Y=y
    Z=z
    nZ=len(Z.T)
    param=np.ones((2*nZ+1))
    #INITIAL STATES
    X0=np.zeros((nZ,1))
    #INITIAL STATE COVARIANCE
    P0=np.eye(nZ)*10**6
    #TRANSITION MATRIX
    A=np.eye(nZ)
    #STATE INNOVATION COVARIANCE MATRIX
    Q=np.zeros((nZ,nZ))  
    #SIGNAL COVARIANCE MATRIX
    H=np.zeros((1,1))    
    #GLOBAL PARAMETERS
    _Q=Q; _H=H; _Y=Y; _Z=Z; _X0=X0; _P0=P0; _A=A; _nZ=nZ; _sr=sr
    # OPTIMIZATION
    out_opt=sp.optimize.minimize(ml_FWD,param,method='BFGS', tol=0.001)
    param=out_opt['x']
    print(out_opt['success'])
    #ESTIMATION WITH OPTIMIZED PARAMETERS
    param=param.reshape(max(param.shape))
    H=np.exp(param[0])
    X0=param[1:nZ+1]
    for i in range(nZ):
        Q[i,i]=sr*H
        P0[i,i]=np.exp(param[nZ+1+i])
    (likely,XOUT,POUT,XSOUT)=kalman_filter(_Y,_Z,X0,P0,A,Q,H)   
    return XOUT,XSOUT

def ml_FWD(param):
    param=param.reshape(max(param.shape))
    _H=np.exp(param[0])
    _X0=param[1:_nZ+1]
    for i in range(_nZ):
        _Q[i,i]=_sr*_H
        _P0[i,i]=np.exp(param[_nZ+1+i])
    (likely,XOUT,POUT,XSOUT)=kalman_filter(_Y,_Z,_X0,_P0,_A,_Q,_H)
    return -likely

def kalman_filter(Y,Z,X0,P0,A,Q,H):
    #Y(T,1) Z(T,nZ) X0(nZ) P0(nZ,nZ) A(nZ,nZ) Q(nZ,nZ) H(1)   
    nZ=len(Z.T)
    nT=len(Y)
    #FOR FILTER
    Zt=np.zeros((1,nZ))
    Xout=np.zeros((nT,nZ))
    Pout=np.zeros((nT,nZ,nZ))
    #FOR SMOOTHER
    XoutS=np.zeros((nT,nZ))
    XtS=np.zeros((nT,nZ))
    Pt1t=np.zeros((nT,nZ,nZ))
    Ptt=np.zeros((nT,nZ,nZ))
    Xtt=np.zeros((nT,nZ))
    Xt1t=np.zeros((nT,nZ))
    #FILTER
    l1=0.0
    l2=0.0
    likely=0.0
    Xt=X0
    Pt=P0
    #FILTER
    for t in range(0,nT,1):
        #estimation step
        Xte=A.dot(Xt)
        Pte=A.dot(Pt).dot(A.T)+Q            
        #update
        Zt[:,:]=Z[t,:]
        vt=Y[t]-Zt.dot(Xte)
        Ft=Zt.dot(Pte).dot(Zt.T)+H
        if np.linalg.det(Ft)==0:
            Ft=Ft+np.random.normal(0,10**-5,Ft.shape)
        Fti=np.linalg.inv(Ft)
        Xt=Xte+Pte.dot(Zt.T).dot(Fti).dot(vt)
        Pt=(np.eye(nZ)-Pte.dot(Zt.T).dot(Fti).dot(Zt)).dot(Pte)
        #stockage
        Xout[t,:]=Xt.T
        Pout[t,:,:]=Pt
        l1=l1+np.log(np.linalg.det(Ft))
        l2=l2+(vt.T).dot(Fti).dot(vt)
        #pour smoother
        Ptt[t,:,:]=Pt
        Xtt[t,:]=Xt.T
        Pt1t[t,:,:]=Pte
        Xt1t[t,:]=Xte.T
    likely=-0.5*nT*np.log(2*math.pi)-0.5*l1-0.5*l2
    #SMOOTHER
    XtS[nT-1,:]=Xtt[nT-1,:]
    XoutS[nT-1,:]=Xtt[nT-1,:]
    for t in range(nT-2,-1,-1):
        if abs(np.linalg.det(Pt1t[t+1,:,:]))==0:
            Pt1t[t+1,:,:]=Pt1t[t+1,:,:]+np.random.normal(0,10**-5,Pt1t[t+1,:,:].shape)
        inverse=np.linalg.inv(Pt1t[t+1,:,:])
        Ft=Ptt[t,:,:].dot(A.T).dot(inverse)
        XtS[t,:]=Xtt[t,:]+Ft.dot(XtS[t+1,:]-Xt1t[t+1,:])
        XoutS[t,:]=XtS[t,:]
    likely=np.reshape(likely,(1))
    return likely/nT,Xout,Pout,XoutS