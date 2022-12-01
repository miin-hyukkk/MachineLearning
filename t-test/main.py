import numpy as np
from scipy import stats

# sample size = 10
N = 10
# x : older adults
# y : younger adults
x = np.array([45, 38, 52, 48, 25, 39, 51, 46, 55, 46])
y = np.array([34, 22, 15, 27, 37, 41, 24, 19, 26, 36])
# calculate the standard deviation
var_x = x.var(ddof=1)
var_y = y.var(ddof=1)
# std
s = np.sqrt((var_x + var_y) / 2)
## Calculate the t-statistics
t = (x.mean() - y.mean()) / (s * np.sqrt(2 / N))
## Compare with the critical t value #degrees of freedom
df = 2 * N - 2
# p value after comparison with the t
p = 1 - stats.t.cdf(t, df=df)
print("t = " + str(t))
print("p = " + str(2 * p))
t2, p2 = stats.ttest_ind(x, y)
print("t = " + str(t2))
print("p = " + str(p2))
