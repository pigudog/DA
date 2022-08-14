from ast import main
# from selectors import EpollSelector
from tkinter.tix import MAIN
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.preprocessing import Normalizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

## sl:satisfaction_level:False:MinMaxScaler,True:StandardScaler
## le:last_evaluation:
## ...
## department:False:labelencode,True:onehotencode
## ...
def hr_preprocessing(sl=False,le=False,npr=False,amh=False,tsc=False,wa=False,pl5=False,dp=False,slr=False,lower_d=False,ld_n=1):
    df = pd.read_csv("HR.csv")
    
    # 1.clean the data
    df.dropna(subset=["satisfaction_level","last_evaluation"])
    df = df[df["satisfaction_level"]<=1][df["salary"]!="nme"]
    
    # 2.find label
    label = df["left"]
    df = df.drop("left",axis=1)
    
    # 3.feature selection
    
    # 4.feature processing
    scaler_lst = [sl,le,npr,amh,tsc,wa,pl5]
    column_lst = ["satisfaction_level","last_evaluation","number_project",\
                "average_montly_hours","time_spend_company","Work_accident",\
                "promotion_last_5years"]
    for i in range(len(scaler_lst)):
        if not scaler_lst[i]:
            df[column_lst[i]] =\
                MinMaxScaler().fit_transform(df[column_lst[i]].values.reshape(-1,1)).reshape(1,-1)[0]
        else:
            df[column_lst[i]]=\
                StandardScaler().fit_transform(df[column_lst[i]].values.reshape(-1,1)).reshape(1,-1)[0]
                

    scaler_lst=[slr,dp]
    column_lst=[ "salary","sales"]
    for i in range(len(scaler_lst)):
        if not scaler_lst[i]:
            if column_lst[i]=="salary":
                df[column_lst[i]]=[map_salary(s) for s in df["salary"].values]
            else:
                df[column_lst[i]]=LabelEncoder().fit_transform(df[column_lst[i]])
                df[column_lst[i]] = MinMaxScaler().fit_transform(df[column_lst[i]].values.reshape(-1, 1)).reshape(1, -1)[0]
        else:
            df=pd.get_dummies(df,columns=[column_lst[i]])
            
    if lower_d:
        return PCA(n_components=ld_n).fit_transform(df.values),label
    return df,label

def map_salary(s):
    d = dict([("low",0),("median",0),({"high",2})])
    return d.get(s,0)

def main():
    print(hr_preprocessing())

if __name__ == "__main__":
    main()   