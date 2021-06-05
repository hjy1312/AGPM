import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from scipy.io import loadmat

a=loadmat('eer.mat')['eer']
b=loadmat('auc.mat')['auc']
print('eer',a.reshape(b.shape[1],1))
print('auc',b.reshape(b.shape[1],1))
import numpy as np
a=a.flatten().reshape(b.shape[1],1)
b=b.flatten().reshape(b.shape[1],1)
print(np.mean(a[-5:,:]))
print(np.mean(b[-5:,:]))
np.savetxt('eer.txt',a,fmt=['%s']*a.shape[1],newline = '\n')

np.savetxt('auc.txt',b,fmt=['%s']*b.shape[1],newline = '\n')

c = np.loadtxt('../acc_cplfw_2.txt')
print(np.mean(c[-5:,0])*100,np.mean(c[-5:,1])*100)
plt.plot(c[:,0], label="FF")
plt.plot(c[:,1], label="FP")
plt.legend(loc='upper right')
plt.xlabel("Epoch")
plt.ylabel("Acc")
plt.show()
filename = './Accuracy_curve.png'
plt.savefig(filename, bbox_inches='tight')

