from classCF import *
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

ratings_base = pd.read_csv('ub.base', sep='\t',
                           names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('ub.test', sep='\t',
                           names=r_cols, encoding='latin-1')

#rate_train = ratings_base.as_matrix()
rate_train = ratings_base.to_numpy()

rate_test = ratings_test.to_numpy()

# indices start from 0
rate_train[:, :2] -= 1
rate_test[:, :2] -= 1

COS = cos_similarity  # method = 1
COS_lib = cosine_similarity  # method = 1
JAC = jaccard_similarity  # method = 2
MSD = msd_similarity  # method = 3
COR = cor_similarity  # method = 4
CPC = cpc_similarity  # method = 5
RJ = rjaccard_similarity

rs = CF(rate_train, k=30, dist_func=MSD, uuCF=1)
# fit(method)
rs.fit(3)

n_tests = rate_test.shape[0]
SE = 0  # squared error
for n in range(n_tests):
    pred = rs.pred(rate_test[n, 0], rate_test[n, 1], normalized=0)
    SE += (pred - rate_test[n, 2])**2

RMSE = np.sqrt(SE/n_tests)
print('User-user CF, RMSE =', RMSE)
