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

COS = cos_similarity  # method = 3
COS_lib = cosine_similarity  # method = 1
JAC = jaccard_similarity  # method = 2
MSD = msd_similarity  # method = 3
COR = cor_similarity  # method = 3
CPC = cpc_similarity  # method = 3
RJ = rjaccard_similarity  # method = 2
# METHOD 4 DÙNG CHO LOOP_USERZZZZZ
rs = CF(rate_train, k=30, dist_func=COS_lib, uuCF=1)
# fit(method)
rs.fit(1)

n_tests = rate_test.shape[0]
SE = 0  # squared error
for n in range(n_tests):
    pred = rs.pred(rate_test[n, 0], rate_test[n, 1], normalized=1)
    print("pred", pred)
    print("rate_test[{}, 2]={}".format(n, rate_test[n, 2]))
    SE += (pred - rate_test[n, 2]) ** 2

RMSE = np.sqrt(SE/n_tests)
print('User-user CF, RMSE =', RMSE)

'''print("rate_train test: ", rate_train)
print("rate_train.shape", rate_train.shape)
print("rate test: ", rate_test)
print("rate_test.shape", rate_test.shape)
'''
