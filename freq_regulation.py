
# coding: utf-8

# In[1]:


import cvxpy as cp
from cvxpy import *
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import time
import pandas as pd


# In[223]:


# see files without loading
sio.whosmat('Reg_signals.mat')


# In[2]:


regD_mat = sio.loadmat('Reg_signals.mat')


# In[3]:


regD_ar = regD_mat["RegD_May07"]
type(regD_ar)
np.shape(regD_ar)[0]


# In[4]:


# plot regD
plt.figure(300, dpi=100)
plt.plot(regD_ar[0:2000])
#plt.show()


# In[6]:


# mobile ESS parameters
T = 300;
# T = np.shape(regD_ar)[0]
# T = 5;
L = 0.02; # MW
E_max = L*T
# E_max = L
# P_max = E_max
P_max = 2*L
# charge is denoted by negative sign
E_init = 0.5*E_max
# E_init = regD[0]
# E_init = 0
# assert E_init <=0, "E_init is positive"
lam_c = 50; # $/MWh
lam_mis = 5 # $/MWh
regD = -regD_ar[0:T]
# T = np.shape(regD)[0]
#  define the subtraction matrix
Add_mat = np.tril(np.ones((T,T)))#, dtype=int))
if np.sign(E_init) == np.sign(regD[0]):
    print('E_init and RegD start with the same sign')
elif E_init == 0:
    print('E_init = 0')
else: print('Different signs')


# In[7]:


# # print(Add_mat*Bt.value[0:5])
# print('Add*Bt: ',np.shape(Add_mat*Bt.value))
# print('Add dot Bt: ',np.shape(np.dot(Add_mat,Bt.value)))
# print('Add@Bt: ',np.shape(Add_mat@Bt.value))
# print('Bt: ',np.shape(Bt.value))


# In[8]:


# Bt is discharge - charge
t0 = time.time()
Bt = cp.Variable((T, 1))
epsilon = cp.Variable((T, 1))
# ob = cp.Variable()
objective = cp.Minimize(sum(abs(epsilon)))
# objective = cp.Minimize(np.max(np.dot(Sub_mat,Xt)))
# objective = cp.Minimize(np.max(np.multiply(Sub_mat,Xt)))
constraints = [abs(Bt-L*regD) <= epsilon,
#                Bt[0] == E_init,
               -P_max <= Bt,
               Bt <= P_max,
              Add_mat@(-Bt) + E_init <= E_max,
              0*E_max <= Add_mat@(-Bt) + E_init]
#               0*E_max <= Add_mat@(-Bt),
#                Add_mat@(-Bt) <= E_max]

prob = cp.Problem(objective, constraints)
prob.solve()  # Returns the optimal value.
t1 = time.time()
print('Elapsed time:',t1-t0)
print("status:", prob.status)
print("optimal value", prob.value)
# print("optimal var", Bt.value)


# In[9]:


fig2 = plt.figure(100, figsize=(9, 6), dpi=200, facecolor='w', edgecolor='k')
plt.plot(Bt.value)
# plt.plot(Bt.value-L*regD)
plt.plot(L*regD)
plt.legend(['Bt','LregD'])
plt.grid(True)
#plt.show()


# In[10]:


mis_pen = lam_mis*np.sum(np.abs(Bt.value - L*regD));
serv_ben = lam_c*L*30 - mis_pen # each 10 time intervals correspond to an hour
print("Mismatch penalty is: ",mis_pen)
print("Benefit is: ",serv_ben)


# In[11]:


fig = plt.figure(200, dpi=150)
fig.subplots_adjust(hspace=0.2, wspace=0.2)
fig.set_figheight(10)
fig.set_figwidth(15)
plt.subplot(221)
plt.plot(Bt.value)
# plt.bar(range(len(Bt.value)),Bt.value)
plt.xlabel('time (s)')
plt.ylabel('MW')
plt.title('Battery charge status b(t)')
plt.grid(True)

plt.subplot(222)
plt.plot(L*regD)
plt.xlabel('time (s)')
plt.ylabel('MW')
plt.title('Regulation Load Lr(t)')
plt.grid(True)

plt.subplot(223)
plt.plot(Bt.value - L*regD)
plt.xlabel('time (s)')
plt.ylabel('MW')
plt.title('b(t)-Lr(t)')
plt.grid(True)

plt.subplot(224)
plt.plot(epsilon.value)
plt.xlabel('time (s)')
# plt.ylabel('MW')
plt.title('Epsilon')
plt.grid(True)

#plt.show()


# In[12]:


plt.figure(400, dpi=150)#, facecolor='w', edgecolor='k')
# plt.plot(Bt.value)
# plt.plot(Bt.value-L*regD)
plt.plot((100/E_max)*(np.cumsum(-Bt.value) + E_init))
plt.title('Battery Charge %')
plt.grid(True)  
#plt.show()


# In[13]:


data1 = (100/E_max)*(np.cumsum(-Bt.value) + E_init)
data2 = L*regD

fig3 = plt.figure(600, dpi=150)

fig3, ax1 = plt.subplots()
fig3.set_figheight(7)
fig3.set_figwidth(12)

color = 'tab:red'
ax1.set_xlabel('time (s)')
ax1.set_ylabel('Battery charge %', color=color)
ax1.plot(data1, color=color)
ax1.tick_params(axis='y', labelcolor=color)
plt.legend(['Bt','LregD'])
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('LregD', color=color)  # we already handled the x-label with ax1
ax2.plot(data2, color=color)
plt.legend(['LregD'])
ax2.tick_params(axis='y', labelcolor=color)
# ax2.annotate('local maximum', xy=(6.28, 1), xytext=(10, 4),
#             arrowprops=dict(facecolor='black', shrink=0.05))

plt.axhline(y=0, color='g', linestyle='-')
plt.grid(True)  

fig3.tight_layout()  # otherwise the right y-label is slightly clipped
#plt.show()


# In[14]:


# dataframe setup
res = {'Epsilon': list(epsilon.value),
       'B(t)': list(Bt.value), 
       'Charge': list(E_init + np.cumsum(-Bt.value)) 
      }
res_df = pd.DataFrame(res)
# res_df


# In[15]:


np.equal(Bt.value[0],-E_init)


# In[16]:


# execute terminal commands
#get_ipython().system('ls ')


# In[17]:


# np.array_equal(np.dot(Add_mat,Bt.value),np.cumsum(Bt.value))
np.array_equal(epsilon.value, Bt.value - L*regD)
np.shape(epsilon.value)

