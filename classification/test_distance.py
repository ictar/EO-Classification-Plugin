k_fold = 4
n_cross = n_train + n_vali
x_cross, y_cross = x[:n_cross], y[:n_cross]
cross_indices = list(range(len(x_cross)))
MSCE_cross = np.zeros()