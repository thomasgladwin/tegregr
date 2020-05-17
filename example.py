import numpy as np
import tegregr

# Params for simulated data
N = 100
k = 3
k_baseline = 6
coeffs_true = np.array([0.1 * n for n in range(k)])
offset_t = 20

# Create arrays
X = np.array([np.random.randn(N) for n in range(k)]).T
X_baseline = np.array([np.random.randn(N) for n in range(k_baseline)]).T
y = np.matmul(X, coeffs_true) + offset_t + np.random.randn(N)

# Run multiple regression
print('Multiple')
Res = tegregr.teg_regression(X, y)
tegregr.teg_report_regr(Res)
print()

# Run hierarchical regression
print('Hierarchical')
Res = tegregr.teg_regression(X, y, X_baseline)
tegregr.teg_report_regr(Res)
print()
