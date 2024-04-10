import numpy as np
import scipy.stats as st

# Outer Fold Nr. 1
# ANN: h = 5, error = [59.43071]
# Regression: lambda = 10.0, error = 59.50130533103987
# Baseline: 89.48244653705346

# Outer Fold Nr. 2
# ANN: h = 6, error = [49.900818]
# Regression: lambda = 10.0, error = 51.51581100258098
# Baseline: 73.02537262760588

# Outer Fold Nr. 3
# ANN: h = 6, error = [98.980934]
# Regression: lambda = 10.0, error = 98.51224800763777
# Baseline: 166.75620173032698

# Outer Fold Nr. 4
# ANN: h = 3, error = [57.099392]
# Regression: lambda = 10.0, error = 57.45939058607159
# Baseline: 91.78555082767885

# Outer Fold Nr. 5
# ANN: h = 7, error = [42.234825]
# Regression: lambda = 10.0, error = 41.153955288368294
# Baseline: 61.09741894831978

#Outer Fold Nr. 6
#ANN: h = 0, error = [117.637085]
#Regression: lambda = 10.0, error = 63.81107787548622
#Baseline: 108.64851682859391

#Outer Fold Nr. 7
#ANN: h = 8, error = [57.354103]
#Regression: lambda = 10.0, error = 63.72424840630876
#Baseline: 103.57777551473637

#Outer Fold Nr. 8
#ANN: h = 5, error = [65.74044]
#Regression: lambda = 10.0, error = 65.05662170432963
#Baseline: 93.77050560476778

#Outer Fold Nr. 9
#ANN: h = 1, error = [93.0748]
#Regression: lambda = 10.0, error = 51.08498641292877
#Baseline: 78.49525862074368

#Outer Fold Nr. 10
#ANN: h = 4, error = [51.160736]
#Regression: lambda = 10.0, error = 51.148801707197464
#Baseline: 78.72193240746337

# E_ann = [59.43, 49.90, 98.98, 57.10, 42.23, 117.64, 57.35, 65.74,93.07, 51.16 ]
# E_reg = [59.50, 51.51, 98.51, 57.46, 41.15, 63.81, 63.72, 65.06, 51.08, 51.14  ]
# E_bas = [89.48, 73.02, 166.75, 91.78, 61.10, 108.65, 103.58, 93.77, 78.50, 78.72]

# z_ann = np.array([[x] for x in E_ann])
# z_reg = np.array([[x] for x in E_reg])
# z_bas = np.array([[x] for x in E_bas])

N_test = 2277 

alpha = 0.05

E_bas = [ 0.509, 0.509, 0.517, 0.513, 0.504, 0.518, 0.544, 0.511, 0.537,0.498]
E_knn = [0.289 ,0.289 ,0.307 ,0.294 ,0.307 ,0.272 ,0.294 ,0.295 ,0.334 ,0.251]
E_log = [ 0.254, 0.254 ,0.316 ,0.276 ,0.268 ,0.263, 0.294, 0.269, 0.247, 0.264]

acc_bas = (sum([1-x  for x in E_bas])/10)*100
acc_knn = (sum([1-x  for x in E_knn])/10)*100
acc_log = (sum([1-x  for x in E_log])/10)*100

print(f"bas. acc. = {acc_bas}%")
print(f"log. acc. = {acc_log}%")
print(f"knn. acc. = {acc_knn}%")

z_knn = np.array([[x] for x in E_knn])
z_log = np.array([[x] for x in E_log])
z_bas = np.array([[x] for x in E_bas])


# Compute confidence interval of z = zA-zB and p-value of Null hypothesis
z = z_log - z_knn 
CI = st.t.interval(
    1 - alpha, len(z) - 1, loc=np.mean(z), scale=st.sem(z)
)  # Confidence interval
print("__")
print(" Log vs KNN")
print(CI)
p = 2 * st.t.cdf(-np.abs(np.mean(z)) / st.sem(z), df=len(z) - 1)  # p-value
print(format(p[0], '.15f'))

z = z_bas - z_knn
CI = st.t.interval(
    1 - alpha, len(z) - 1, loc=np.mean(z), scale=st.sem(z)
)  # Confidence interval
print("__")
print(" Bas vs KNN")
print(CI)
p = 2 * st.t.cdf(-np.abs(np.mean(z)) / st.sem(z), df=len(z) - 1)  # p-value
print(format(p[0], '.15f'))

z =  z_log - z_bas
CI = st.t.interval(
    1 - alpha, len(z) - 1, loc=np.mean(z), scale=st.sem(z)
)  # Confidence interval
print("__")
print(" Log vs Bas")
print(CI)
p = 2 * st.t.cdf(-np.abs(np.mean(z)) / st.sem(z), df=len(z) - 1)  # p-value
print(format(p[0], '.15f'))


