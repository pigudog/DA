from ast import main
# from selectors import EpollSelector
from tkinter.tix import MAIN
from xmlrpc.server import SimpleXMLRPCDispatcher
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.preprocessing import Normalizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import os
os.environ["PATH"]+=os.pathsep+"D:/program/Graphviz/bin/"
import pydotplus

## 请忽略这一条，这只是一个用来测试branch的文字

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

def hr_modeling(features,label):
    from sklearn.model_selection import train_test_split
    f_v = features.values
    f_names = features.columns.values
    l_v = label.values
    X_tt,X_validation,Y_tt,Y_validation = train_test_split(f_v,l_v,test_size=0.2)
    X_train,X_test,Y_train,Y_test = train_test_split(X_tt,Y_tt,test_size = 0.25)
    print(len(X_train),len(X_validation),len(X_test))
    
    

    from sklearn.neighbors import NearestNeighbors,KNeighborsClassifier
    from sklearn.metrics import accuracy_score,recall_score,f1_score
    from sklearn.naive_bayes import GaussianNB,BernoulliNB
    from sklearn.tree import DecisionTreeClassifier,export_graphviz
    from six import StringIO
    from sklearn.svm import SVC
    
    
    models = []
    # models.append(("KNN",KNeighborsClassifier(n_neighbors=3)))
    # models.append(("GussianNB",GaussianNB()))
    # models.append(("BernoulliNB",BernoulliNB()))
    # models.append(("DecisionTree",DecisionTreeClassifier()))
    models.append(("SVM Classifier",SVC()))
    

    list3 = ["test:","validation:","test:"]
    for clf_name,clf in models:
        clf.fit(X_train,Y_train)
        xy_lst = [(X_train,Y_train),(X_validation,Y_validation),(X_test,Y_test)]
        for i in range(len(xy_lst)):
            X_part = xy_lst[i][0]
            Y_part = xy_lst[i][1]
            Y_pred = clf.predict(X_part)
            print(list3[i])
            print(clf_name,"-ACC",accuracy_score(Y_part,Y_pred))
            print(clf_name,"-REC",recall_score(Y_part,Y_pred))
            print(clf_name,"-F1",f1_score(Y_part,Y_pred))
            if clf_name == "DecisionTree":
                dot_data = export_graphviz(clf,out_file=None,
                                        feature_names=f_names,
                                        class_names=["NL","L"],
                                        filled=True,
                                        rounded=True,
                                        special_characters=True)
                graph = pydotplus.graph_from_dot_data(dot_data)
                graph.write_pdf("dt_tree.pdf")  
    


def main():
    features,label = hr_preprocessing()
    hr_modeling(features,label)

if __name__ == "__main__":
    main()   