import pandas as pd
import torch
import numpy as np
import glob


import torch_geometric
import torch_geometric.data as geom_data

from torch_geometric.data import Data

aa=['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
aa_dict= dict([(aa[i], i) for i in range(0,len(aa))])


def prep_for_graph_features(df):
#if I have time later, I would consider going in and loading in coordinates for all atoms
#but I probably will not tbh
    
    site=[]
    resid=[]  #this will be one hot encoded later
    coord_array=[]
    x=[]
    y=[]
    z=[]
    num_C=[]
    num_N=[]
    num_O=[]
    num_S=[]

    for item in df.resid_num.unique():
        mydf=df.loc[df.resid_num==item]

        atom_list=mydf.atom.to_list()

        just_elem=[i[0] for i in mydf.atom.to_list()]
        num_C.append(just_elem.count('C'))
        num_N.append(just_elem.count('N'))
        num_O.append(just_elem.count('O'))
        num_S.append(just_elem.count('S'))

                     
        site.append(mydf.resid_num.to_list()[0])
        resid.append(aa_dict.get(mydf.wildtype.to_list()[0]))
        atom_list.append(mydf.atom.to_list())
        
        CA_only=mydf.loc[df['atom']=='CA']

        coord_array.append(np.array(CA_only.loc[:,['x','y','z']]))
        x.append(CA_only.loc[:,'x'].to_list()[0])
        y.append(CA_only.loc[:,'y'].to_list()[0])
        z.append(CA_only.loc[:,'z'].to_list()[0])
                     
    
    prepped_df=pd.DataFrame({'site':site,
                             'resid':resid,
                             'coord_array':coord_array,
                             'x_coord':x,
                             'y_coord':y,
                             'z_coord':z,
                             'num_C':num_C,
                             'num_N':num_N,
                             'num_O':num_O,
                             'num_S':num_S})

    return prepped_df


def combining_ensembles(ens_dir_list):
    
    pdb1_df=pd.DataFrame()
    pdb2_df=pd.DataFrame()
    ddg_df=pd.DataFrame()
    
    for i in ens_dir_list:
    
        this_ens=i.split('/')[-2]
        
        df1=prep_for_graph_features(pd.read_csv(glob.glob(i+'*pdb1*csv')[0]))
        df1['ens']=this_ens
        pdb1_df=pd.concat([pdb1_df,df1],ignore_index=True)
        
        df2=prep_for_graph_features(pd.read_csv(glob.glob(i+'*pdb2*csv')[0]))
        df2['ens']=this_ens
        pdb2_df=pd.concat([pdb2_df,df2],ignore_index=True)
        
    
        ddgs=pd.read_csv(glob.glob(i+'*ddgs.csv')[0])
        ddgs['ens']=this_ens
        ddg_df=pd.concat([ddg_df,ddgs],ignore_index=True)

    return pdb1_df, pdb2_df, ddg_df


def get_edge_info(df):

    edges=[]
    edge_features=[]
    
    for my_ens in df.ens.unique():

        my_df=df.loc[df.ens == my_ens]

        start_idx=my_df.index[0]
        end_idx=my_df.index[-1]
    
        #pdb file coordinate numbers are angstroms
        #I am creating edges between all alpha carbons within 10a of each other

        #0 for near each other, 1 for backbone bond
        for idx1,rows1 in my_df.iterrows():
        
            p1=my_df.loc[idx1,'coord_array']
        
            if idx1 == start_idx:
                edges.append([idx1,idx1+1])
                edge_features.append(1)
        
            elif idx1 == end_idx:
                edges.append([idx1,idx1-1])
                edge_features.append(1)
                
            else:
                edges.append([idx1, idx1-1])
                edges.append([idx1,idx1+1])
                edge_features.extend([1,1])
                
        
            for idx2,rows2 in my_df.iterrows():
        
                if idx1 != idx2:
        
                    p2=my_df.loc[idx2,'coord_array']
            
                    d = np.linalg.norm(p1 - p2)
        
                    if d<=10:
        
                        edges.append([idx1,idx2])
                        
                        edge_features.append(0)


    return edges, edge_features


def load_geom_data(df):

    mydata=Data()

    mydata.pos=torch.tensor(df[:][['x_coord','y_coord','z_coord']].values)
    mydata.x=torch.tensor(df[:][['num_C','num_N','num_O','num_S']].values)
    mydata.resid=torch.nn.functional.one_hot(torch.Tensor(df.resid).long(), num_classes=20)

    edges, edge_features=get_edge_info(df)


    mydata.edge_index=torch.transpose(torch.tensor(edges),0,1)
    mydata.edge_attrs=torch.tensor(edge_features)

    return mydata
    
