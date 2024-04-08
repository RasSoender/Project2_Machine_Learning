import numpy as np
import scipy.stats as st


E_ann = [59.43, 49.90, 98.98, 57.10, 42.23, 117.64, 57.35, 65.74,93.07, 51.16 ]
E_reg = [59.50, 51.51, 98.51, 57.46, 41.15, 63.81, 63.72, 65.06, 51.08, 51.14  ]
E_bas = [89.48, 73.02, 166.75, 91.78, 61.10, 108.65, 103.58, 93.77, 78.50, 78.72]

N_test = 2277 

alpha = 0.05

z_ann = np.array([[x] for x in E_ann])
z_reg = np.array([[x] for x in E_reg])
z_bas = np.array([[x] for x in E_bas])

# Compute confidence interval of z = zA-zB and p-value of Null hypothesis
z = z_reg - z_ann 
CI = st.t.interval(
    1 - alpha, len(z) - 1, loc=np.mean(z), scale=st.sem(z)
)  # Confidence interval
print("__")
print(" Reg vs ANN")
print(CI)
p = 2 * st.t.cdf(-np.abs(np.mean(z)) / st.sem(z), df=len(z) - 1)  # p-value
print(format(p[0], '.10f'))

z = z_bas - z_ann
CI = st.t.interval(
    1 - alpha, len(z) - 1, loc=np.mean(z), scale=st.sem(z)
)  # Confidence interval
print("__")
print(" Bas vs ANN")
print(CI)
p = 2 * st.t.cdf(-np.abs(np.mean(z)) / st.sem(z), df=len(z) - 1)  # p-value
print(format(p[0], '.10f'))

z =  z_reg - z_bas
CI = st.t.interval(
    1 - alpha, len(z) - 1, loc=np.mean(z), scale=st.sem(z)
)  # Confidence interval
print("__")
print(" Reg vs Bas")
print(CI)
p = 2 * st.t.cdf(-np.abs(np.mean(z)) / st.sem(z), df=len(z) - 1)  # p-value
print(format(p[0], '.10f'))


