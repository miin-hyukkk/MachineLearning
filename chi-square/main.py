from scipy.stats import chi2_contingency
from scipy.stats import chi2
import numpy as np

# contingency_table
contingency_table = [
    [200, 150, 50],
    [250, 300, 50]
]
print(contingency_table)
stat, p, dof, expected = chi2_contingency(contingency_table)
print('dof=%d' % dof)
print(expected)
# interpret test statistic
prob = 0.95
critical = chi2.ppf(prob, dof)
print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
if abs(stat) >= critical:
    print('Dependent (reject H0)')
else:
    print('Independent (fail to reject H0)')
# interpret p value
alpha = 1.0 - prob
print('significance=%.3f, p=%.3f ' % (alpha, p))
if p <= alpha:
    print('Dependent (reject H0)')
else:
    print('Independent (fail to reject H0)')
