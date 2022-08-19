from turtle import circle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles,make_blobs,make_moons


n_samples = 1000
circles = make_circles(n_samples=n_samples,factor=0.5,noise=0.05)
moons = make_moons(n_samples=n_samples,noise=0.05)
blobs = make_blobs(n_samples=n_samples,random_state=8)

random_data = np.random.rand(n_samples,2),None

colors="bgrcmyk"
data = [circles,moons,random_data]

models = [("None",None)]
f = plt.figure()
for inx,clt in enumerate(models):
    clt_name,clt_entity = clt
    for i,dataset in enumerate(data):
        X,Y = dataset
        if not clt_entity:
            clt_res = [0 for item in range(len(X))]
        f.add_subplot(len(models),len(data),inx*len(data)+i+1)
        [plt.scatter(X[p,0],X[p,1],color=colors[clt_res
                                                    [p]]) for p in range(len(X))]
    plt.show()
            
            
