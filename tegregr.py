import numpy as np
from scipy import stats

def get_F(X, y, coeffs):
    N, k = np.shape(X)
    # Calculate statistics
    pred = np.matmul(X, coeffs)
    res = y - pred
    R2 = np.var(pred) / np.var(y)
    # F-test of overall model fit
    df1 = k - 1
    df2 = N - 1 - df1
    SSM = sum((pred - np.mean(pred))**2)
    MSM = SSM / df1
    SSE = sum((res - np.mean(res))**2)
    MSE = SSE / df2
    F = MSM/MSE
    F_p = 1 - stats.f.cdf(F, df1, df2)
    ErrVar = np.var(res)
    resss = sum(res**2)
    return F, F_p, R2, df1, df2, ErrVar, resss

def get_coeff_test(X, resss, coeffs):
    N, k_off = np.shape(X)
    k = k_off - 1 # k is number of redictors
    t_vec = []
    p_vec = []
    se_vec = []
    MSE = resss / (N - k - 1)
    invXtX = np.linalg.inv(np.matmul(X.T, X))
    for ik in range(len(X.T)):
        se = np.sqrt(MSE * invXtX[ik][ik])
        if se > 0:
            se_vec.append(se)
            t = coeffs[ik]/se
            t_vec.append(t)
            df_t = N - k - 1
            p = 1 - stats.t.cdf(t, df_t)
            p = 2 * np.min([p, 1-p])
            p_vec.append(p)
        else:
            t_vec.append(0)
            p_vec.append(1)
    return t_vec, p_vec, df_t, se_vec

def hierarchical(baseline_X, y, df1, ErrVar):
    N, k_baseline = np.shape(baseline_X)
    Res_baseline = teg_regression(baseline_X, y)
    F_baseline = Res_baseline['F']
    df1_baseline = Res_baseline['df1']
    df2_baseline = Res_baseline['df2']
    ErrVar_baseline = Res_baseline['ErrVar']
    Delta_df1 = df1 - df1_baseline
    Delta_df2 = N - df1
    Delta_F = ((ErrVar_baseline * (N-1) - ErrVar * (N-1)) / Delta_df1) / (ErrVar * (N-1) / Delta_df2)
    Delta_p = 1 - stats.f.cdf(Delta_F, Delta_df1, Delta_df1)
    return Delta_F, Delta_p, Delta_df1, Delta_df2

def teg_regression(X, y, baseline_X = []):
    # X is a (N, k) NumPy array
    # y is a (N,) NumPy array
    # baseline_X is an optional baseline model
    #
    # X and baseline_X should not include a ones column.
    if len(np.shape(X)) == 1:
        X = np.reshape(X, (len(X), 1))
    if len(baseline_X) > 0:
        X = np.hstack([X, baseline_X])
    N, k = np.shape(X)
    Inter = np.ones((N, 1))
    X = np.hstack([X, Inter])
    # Get least-squares coefficients
    Fit = np.linalg.lstsq(X, y, rcond=None)
    coeffs = Fit[0]
    # Get model fit
    F, F_p, R2, df1, df2, ErrVar, resss = get_F(X, y, coeffs)
    # T-tests per predictor
    t_vec, p_vec, df_t, se_vec = get_coeff_test(X, resss, coeffs)
    # Hierarchical
    if len(baseline_X) > 0:
        Delta_F, Delta_p, Delta_df1, Delta_df2 = hierarchical(baseline_X, y, df1, ErrVar)
    else:
        Delta_F = 0
        Delta_p = 1
        Delta_df1 = 0
        Delta_df2 = 0
    # Return stats
    return ({'b':coeffs, 'R2':R2, 
        'df1':df1, 'df2':df2, 'F':F, 'F_p':F_p,
        't':t_vec, 't_p':p_vec, 'df_t':df_t, 'se_vec':se_vec, 'Delta_F':Delta_F, 'Delta_p':Delta_p, 'Delta_df1':Delta_df1, 'Delta_df2':Delta_df2, 'ErrVar':ErrVar})

def teg_report_regr(Res):
    print('R2 = ' + str(np.around(Res['R2'], 3)) + ', F(' + str(np.around(Res['df1'], 3)) + ', ' + str(np.around(Res['df2'], 3)) + ') = ' + str(np.around(Res['F'], 3)) + ', p = ' + str(np.around(Res['F_p'], 3)))
    if Res['Delta_df1'] > 0:
        print('Delta F(' + str(np.around(Res['Delta_df1'], 3)) + ', ' + str(np.around(Res['Delta_df2'], 3)) + ') = ' + str(np.around(Res['Delta_F'], 3)) + ', p = ' + str(np.around(Res['Delta_p'], 3)))
    for ik in range(len(Res['b']) - 1):
        print('b[' + str(ik) + '] = ' + str(np.around(Res['b'][ik], 3)) + ', se(b) = ' + str(np.around(Res['se_vec'][ik], 3)) + ', t(' + str(np.around(Res['df_t'], 3)) + ') = ' + str(np.around(Res['t'][ik], 3)) + ', p = ' + str(np.around(Res['t_p'][ik], 3)))
    print('Offset = ' + str(np.around(Res['b'][-1], 3)))

def create_correlated_variable(X, r):
    if len(np.shape(X)) == 1:
        X = np.reshape(X, (len(X), 1))
    N, k = np.shape(X)
    Y_init = np.random.randn(N)
    Res = teg_regression(X, Y_init)
    Y_pred = np.matmul(np.hstack([X, np.ones((N, 1))]), Res['b'])
    Y_resid = Y_init - Y_pred
    Y_pred = stats.zscore(Y_pred)
    Y_resid = stats.zscore(Y_resid)
    Y_correlated = r * Y_pred + np.sqrt(1 - r**2) * Y_resid
    return Y_correlated
