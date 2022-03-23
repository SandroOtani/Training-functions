from tqdm import tqdm
import numpy as np
import pandas as pd

def KS(df,train_size):
    '''
    Knnard-stone function, where the first point to be chosen is the closest to the mean
    Code adapted from the MATLAB version of kenston writed by Wen Wu
    Input:
        df: dataframe containing the data to be separated into training and testing sets
        train_size: percentage of samples to be chosen for training (value between 0 and 1)
    Output:
        train_index: training set samples indexes
        test_index: test set samples indexes
    Warning: 
        Before using the function, it is recommended to use the dataframe with the reseted indexes (df.reset_index(drop=True))
        For classification problems, this function does not distinguish between classes, so it is 
    recommended that samples from different classes are separated into different dataframes before applying this function.
    
    Sandro K. Otani 09/2021
    Sílvia Claudino Martins Gomes
    Otávio Anovazzi 
    '''
    index = df.index
    df = df.reset_index(drop=True)
    df = df.T.reset_index(drop=True).T
    lin = df.shape[0]
    col = df.shape[1]
    dic={}
    num_obj = round(train_size*lin,0)
    
    me = np.mean(df)
    df_me = df - pd.DataFrame(np.tile(me,(lin,1)))
    k = 0
    a = list()
    ind = list()
    # Encontrando o ponto mais próximo da média
    for k in range(lin):
        dp = np.dot(df_me.iloc[k,::],df_me.iloc[k,::])
        if k==0:
            min_mean=dp
            first=k
        if dp<min_mean:
            min_mean=dp
            first=k
    ind.append(first)
    # Encontrando o ponto mais distante do primeiro
    df2 = df - pd.DataFrame(np.tile(df.iloc[ind,::],(lin,1)))
    k = 0
    a = list()
    for k in range(lin):
        dp = np.dot(df2.iloc[k,::],df2.iloc[k,::])
        if k==0:
            max_far=dp
            second=k
        if dp>max_far:
            max_far=dp
            second=k
    ind.append(second)
    # Encontrando os outros pontos
    for a in tqdm(np.arange(2,num_obj,1)):
        
        list_ind = np.arange(0,lin,1).tolist()
        res = [i for i in list_ind if i not in ind]   
        df3 = df.drop(ind,axis=0)
        ni = 0
        vmin = list()
        dpl = np.zeros(len(ind))
        ind_i=0
        for ni in res:
            for k in range(len(ind)):
                if str(ni)+'-'+str(ind[k]) in dic.keys():
                    if k==0:
                        min_dpl=dic[str(ni)+'-'+str(ind[k])]
                    else:
                        dpl=dic[str(ni)+'-'+str(ind[k])]
                        if dpl<min_dpl:
                            min_dpl=dpl
                elif str(ind[k])+'-'+str(ni) in dic.keys():
                    if k==0:
                        min_dpl=dic[str(ind[k])+'-'+str(ni)]
                    else:
                        dpl=dic[str(ind[k])+'-'+str(ni)]
                        if dpl<min_dpl:
                            min_dpl=dpl
                else:
                    if k==0:
                        t = df.iloc[ni,::]-df.iloc[ind[k],::]
                        min_dpl=np.dot(t,t)
                        dic[str(ind[k])+'-'+str(ni)]=min_dpl
                    else:
                        t = df.iloc[ni,::]-df.iloc[ind[k],::]
                        dpl=np.dot(t,t)
                        if dpl<min_dpl:
                            min_dpl=dpl
                        dic[str(ind[k])+'-'+str(ni)]=dpl
            if ni==res[0]:
                max_mindpl=min_dpl
                ind_res=ind_i
            if min_dpl>max_mindpl:
                max_mindpl=min_dpl
                ind_res=ind_i
            ind_i+=1
        ind.append(res[ind_res]) 
    res = [i for i in list_ind if i not in ind]
    train_index = np.asarray(index[ind])
    test_index = np.asarray(index[res])
    train_index.sort()
    test_index.sort()
    return train_index, test_index