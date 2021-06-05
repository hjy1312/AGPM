import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from scipy.io import loadmat

a=loadmat('std_eer.mat')['std_eer']
b=loadmat('std_auc.mat')['std_auc']
print('std_eer',a.reshape(b.shape[1],1))
print('std_auc',b.reshape(b.shape[1],1))
import numpy as np
a=a.flatten().reshape(b.shape[1],1)
b=b.flatten().reshape(b.shape[1],1)
print(np.mean(a[-5:,:]))
print(np.mean(b[-5:,:]))
np.savetxt('std_eer.txt',a,fmt=['%s']*a.shape[1],newline = '\n')

np.savetxt('std_auc.txt',b,fmt=['%s']*b.shape[1],newline = '\n')

c = np.loadtxt('../std_cfp.txt')
print(np.mean(c[-5:,0]),np.mean(c[-5:,1]))
plt.plot(c[:,0], label="FF")
plt.plot(c[:,1], label="FP")
plt.legend(loc='upper right')
plt.xlabel("Epoch")
plt.ylabel("Acc")
plt.show()
filename = './std_Accuracy_curve.png'
plt.savefig(filename, bbox_inches='tight')

