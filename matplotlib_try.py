import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(np.random.randn(1000).cumsum(),'k',label='one')  #label标签用于添加图例
ax.annotate("Important value", xy = (55,20), xycoords='data',   #添加注释的方法
         xytext=(5, 38),
         arrowprops=dict(arrowstyle='->'))
plt.show()