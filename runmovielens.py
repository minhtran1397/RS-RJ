from classCF import *
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

ratings_base = pd.read_csv('ub.base', sep='\t',
                           names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('ub.test', sep='\t',
                           names=r_cols, encoding='latin-1')

# rate_train = ratings_base.as_matrix()
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
rs = CF(rate_train, k=20, dist_func=COS_lib, uuCF=1)
# fit(method)
rs.fit(1)

n_tests = rate_test.shape[0]
SE = 0  # squared error
pred_arr = np.array([])
real_rate_arr = np.array([])
for n in range(n_tests):
    pred = rs.pred(rate_test[n, 0], rate_test[n, 1], normalized=1)
    pred_arr = np.append(pred_arr, [pred], axis=0)
    real_rate_arr = np.append(real_rate_arr, [rate_test[n, 2]], axis=0)
    # print("pred", [pred])
    # prireal_rate_arrnt("rate_test[{}, 2]={}".format(n, rate_test[n, 2]))
    # Lưu mảng pred
    # kt độ 9xacs trên mảng pred với mảng
    # SE += (pred - rate_test[n, 2]) ** 2
print("pred_arr", pred_arr)
print("real_rate_arr", real_rate_arr)
# RMSE = np.sqrt(SE/n_tests)
RMSE = sqrt(((pred_arr-real_rate_arr)**2).sum()/n_tests)
print('User-user CF, RMSE =', RMSE)
# F1 = precision_recall_fscore_support( real_rate_arr, pred_arr.round(), average = 'macro', zero_division = 1)
F1 = f1_score(real_rate_arr, pred_arr.round(), labels=[1, 2, 3, 4, 5],
              average=None, zero_division=1)
print("f1_score nè:", F1)
rc = recall_score(real_rate_arr, pred_arr.round(), labels=[1, 2, 3, 4, 5],
                  average=None)
print("recall_score nè:", rc)
ps = precision_score(real_rate_arr, pred_arr.round(), labels=[1, 2, 3, 4, 5],
                     average=None)
print("precision_score nè:", ps)
'''print("rate_train test: ", rate_train)
print("rate_train.shape", rate_train.shape)
print("rate test: ", rate_test)
print("rate_test.shape", rate_test.shape)
'''

precision, recall, thresholds = precision_recall_curve(
    real_rate_arr, pred_arr, pos_label=1)
print("precision", precision)
print("recall", recall)
print("thresholds", thresholds)

'''thres4 = np.array([])
for th in range(thresholds):
    if (thresholds[th] == 4):
        thres4 = np.append[th]
print("thres4", thres4)
'''
