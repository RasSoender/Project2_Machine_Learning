import numpy as np
import scipy.stats as st


# h = 7, 8, 4, 3, 7, 6, 4, 5, 4, 5

E_ann = [47.987465,58.737244, 44.881897, 52.0765,78.260574,54.07419,45.0038,56.13082,55.896004,59.608658  ]
E_reg = [52.78281859923111, 63.93365979871374, 48.76998871673114,54.91268417878222,85.78698729722434,58.16021726443172,49.51847868719804,64.46475864148503,58.31044288992146,64.61070696442997 ]
E_bas = [85.77172868280628,92.15008441807662, 81.36666025193003, 74.60650686393105, 128.54807536299492, 88.2608036856956,83.64845160366245,106.2314123281871,112.70160325759224,91.13166614058808]

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


