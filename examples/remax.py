import numpy as np
import matplotlib.pylab as plt
from scipy.stats import lognorm, norm

## Hidden states definition
# nb_Z, s = 10, 0.25  #Example 0
# nb_Z, s = 10, 1     #Example 1
# nb_Z, s = 50, 1     #Example 2
nb_Z, s = 10, 0.05    #Example 3
# nb_Z, s = 500,1     #Example 4

# nb_Z = 100 #Number of variables

mZ = [] #Initialize mean & std of hidden units
sZ = []
for i in range(nb_Z):
    mZ.append(-1)
    sZ.append(s)
mZ[0] = 1
# sZ[0] = 1
mZ = np.array(mZ)
sZ = np.array(sZ)
s2Z = sZ**2

## MCS verification
Z_s = np.random.normal(loc=mZ,scale=sZ,size=(1000000,nb_Z))
# M_s = np.maximum(1e-12,Z_s)
M_s = np.log(1+np.exp(Z_s))
# idx_0 = np.where(np.all(M_s<=0,axis=1))
# M_s = np.delete(M_s,idx_0[0],0)

Msum = sum(M_s.transpose())
A_s = np.divide(M_s,Msum[:,np.newaxis])
mA_s = np.mean(A_s,axis=0)
sA_s = np.std(A_s,axis=0)

## Compute empiriacl covariance between Z and A
cov_matrix = np.cov(Z_s.T, A_s.T, rowvar=True)
cov_ZA = cov_matrix[:nb_Z, nb_Z:]
cov_diag_emp = np.diag(cov_ZA)

# print('Empirical mean(A) = ',mA_s)
# print('Empirical Var(A) = ',sA_s**2)
# print('Empirical Cov(Z,A) = ',cov_diag_emp)


## mReLU activation
# alpha = mZ/sZ
# mM = mZ * norm.cdf(alpha) + sZ * norm.pdf(alpha)
# s2M = -(mM**2) + 2*mM*mZ - mZ * sZ * norm.pdf(alpha) + (s2Z - mZ**2) * norm.cdf(alpha)
# s2M = np.maximum(1e-12,s2M)
# sM = np.sqrt(s2M)

# cov_M_Z = norm.cdf(alpha) * s2Z

## Softplus activation
# mu_a[col] = logf(1 + expf(mu_z[col]));
# tmp = 1 / (1 + expf(-mu_z[col]));
# jcb[col] = tmp;
# var_a[col] = tmp * var_z[col] * tmp;
mM = np.log(1+np.exp(mZ))
tmp = 1/(1+np.exp(-mZ))
s2M = tmp * s2Z * tmp
sM = np.sqrt(s2M)

cov_M_Z = tmp * s2Z


# mM = np.maximum(1e-12,mM)

## lnM = log(M_i)
s2lnM = np.log(1+s2M/mM**2)
# slnM = np.sqrt(s2lnM)
slnM = s2lnM
mlnM = np.log(mM)-0.5*s2lnM
cov_M_lnM = s2lnM*mM

## \tilde{M} = sum(M_i)
mM_sum = sum(mM)
s2M_sum = sum(s2M)
# sM_sum = np.sqrt(s2M_sum)
sM_sum = s2M_sum

# mM_sum = np.maximum(1e-12,mM_sum)

## ln\tilde{M} = log(\tilde{M}_i)
s2lnM_sum = np.log(1+s2M_sum/mM_sum**2)
slnM_sum = np.sqrt(s2lnM_sum)
mlnM_sum = np.log(mM_sum)-0.5*s2lnM_sum
cov_lnM_lnM_sum = np.log(1+s2M/mM/mM_sum)


## \check{A}_i = lnM_i - ln\tilde{M}
mlnA = mlnM - mlnM_sum
s2lnA = s2lnM + s2lnM_sum - 2*cov_lnM_lnM_sum
slnA = np.sqrt(s2lnA)

cov_lnA_lnM = s2lnM - cov_lnM_lnM_sum
cov_lnA_M = cov_lnA_lnM * mM

## A_i = normal
mA = np.exp(mlnA+1/2*s2lnA)
s2A = mA**2*(np.exp(s2lnA)-1)
sA = np.sqrt(s2A)

## Covariance between Z and A
cov_Z_A = mA * cov_lnA_M / s2M * cov_M_Z

J = cov_Z_A / s2Z
J_empirical = cov_diag_emp / s2Z

print('Empirical vs Theoretical ratio:')
print(np.column_stack((J_empirical / sA_s**2, J / s2A)))


# print('mean(A) = ', mA)
# print('Var(A) = ', s2A)
# print('Cov(Z,A) = ',cov_Z_A)

# ===================== Results =====================
print("Empirical vs Theoretical Mean(A):")
print(np.column_stack((mA_s, mA)))

print("\nEmpirical vs Theoretical Var(A):")
print(np.column_stack((sA_s**2, s2A)))

print("\nEmpirical vs Theoretical Cov(Z, A):")
print(np.column_stack((cov_diag_emp, cov_Z_A)))