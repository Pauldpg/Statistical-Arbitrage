# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 13:07:02 2024

@author: gmona
"""
import numpy as np
import scipy as scipy
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import calinski_harabasz_score
from scipy.cluster.hierarchy import linkage, fcluster

def gap_stat(_DIST, _link, _data, _dist, _clusters, _nsim):    
    #stat de dispersion intra
    clusteroids_c=np.zeros((_data.shape[0],max(_clusters)))
    centroids_c=np.zeros((_data.shape[0],max(_clusters)))
    inertia_i=0
    wss=0
    bss=0
    tmean=np.mean(_data,axis=1)
    for i in np.arange(1,max(_clusters)+1,1):
        dist_cl=squareform(_dist)
        dist_cl=dist_cl[:,_clusters==i]
        dist_cl=dist_cl[_clusters==i,:]
        #CLUSTEROIDS+INERTIE INTRA
        data_c=_data.T[_clusters==i,:].T
        mn_c,idx_c = min((np.mean(dist_cl,axis=0)[j],j) for j in range(len(np.mean(dist_cl,axis=0))))
        clusteroids_c[:,i-1]=data_c[:,idx_c]
        #CENTROID
        centroids_c[:,i-1]=np.mean(data_c,axis=1)
        #WSS
        wss=wss+np.sum((data_c-centroids_c[:,i-1:i])**2)
        bss=bss+data_c.shape[1]*np.sum((centroids_c-tmean[0:1])**2)
        #INERTIA
        inertia_i=inertia_i+np.sum(dist_cl)/(2*dist_cl.shape[1])      
    
    #SIMULATIONS GAP STAT
    #(pca_comp, pca_var, pca_loadings, pca_fit)=pca(data,5)
    inertia_i_sim=np.zeros((_nsim,1))
    for i in np.arange(1,_nsim+1,1):
        data_sim=np.dot(np.diag(np.max(_data,axis=1)-np.min(_data,axis=1)),np.random.rand(len(_data),_data.shape[1]))+np.reshape(np.min(_data,axis=1),(_data.shape[0],1))
        dist_sim=pdist(data_sim.T,_DIST)
        output_sim = linkage(dist_sim,method=_link)
        clusters_sim=fcluster(output_sim,max(_clusters),criterion='maxclust')
        dist_cl_sim1=squareform(dist_sim)
        for j in np.arange(1,max(clusters_sim)+1,1):
            dist_cl_sim=dist_cl_sim1[:,clusters_sim==j]
            dist_cl_sim=dist_cl_sim[clusters_sim==j,:]
            #CLUSTEROIDS+INERTIE INTRA
            mn_c,idx_c = min((np.mean(dist_cl_sim,axis=0)[jj],jj) for jj in range(len(np.mean(dist_cl_sim,axis=0))))
            inertia_i_sim[i-1]=inertia_i_sim[i-1]+np.sum(dist_cl_sim)/(2*dist_cl_sim.shape[1])
        inertia_i_sim[i-1]=inertia_i_sim[i-1]
    #calcul output
    gap_stat=np.mean(np.log(inertia_i_sim))-np.log(inertia_i)
    sd_gap_stat=((1+1/_nsim)**0.5)*np.std(np.log(inertia_i_sim))
    #seuil_stat=np.percentile(inertia_i_sim,1)
    seuil_stat=np.min(inertia_i_sim)    
    return gap_stat,sd_gap_stat,centroids_c,inertia_i,inertia_i_sim,seuil_stat