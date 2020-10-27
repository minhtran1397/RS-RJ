from classCF import *
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_fscore_support
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

COS = cos_similarity  # method =4
COS_lib = cosine_similarity  # method = 1
JAC = jaccard_similarity  # method = 2
MSD = msd_similarity  # method = 4
MSDPP = msd_similarity_pp  # method = 4
COR = cor_similarity  # method = 4
CPC = cpc_similarity  # method = 4
RJ = rjaccard_similarity  # method = 2
RJMSD = rjmsd_similarity  # method = 5
RJURP = rjurp_similarity  # method = 3
AMSD = amsd_similarity  # method = 5
SRJ = srj_similarity  # method = 3
PIP = pip_similarity  # method = 7
RPB = rpb_similarity  # method = 4
RATEJRPB = rating_jaccard_rpb_similarity  # method = 5
RJRPB = rj_rpb_similarity  # method = 5
MNRJRPB = mnrj_rpb_similarity  # method = 5
MNRJ_UOD = mnrj_uod_similarity  # method = 5
JACLMH = jaclmh_similarity  # method = 4
NHSM = nhsm_similarity  # method = 7
TMJ = tmj_similarity  # method = 5
# METHOD 4 DÙNG CHO LOOP_USERZZZZZ
rs = CF(rate_train, k=5, dist_func=COS_lib, uuCF=1)
rs.fit(1)
n_tests = rate_test.shape[0]
pred_arr = np.array([])
real_rate_arr = np.array([])
thres_arr = np.array([])
for n in range(n_tests):
    pred, thres = rs.pred(rate_test[n, 0], rate_test[n, 1], normalized=1)
    pred_arr = np.append(pred_arr, [pred], axis=0)
    thres_arr = np.append(thres_arr, [thres], axis=0)
    real_rate_arr = np.append(real_rate_arr, [rate_test[n, 2]], axis=0)
print("pred_arr", pred_arr.shape)
print("real_rate_arr", real_rate_arr.shape)
print("thres_arr", thres_arr.shape)
print("max pred", np.max(pred_arr))
print("min Pred", np.min(pred_arr))
# RMSE = np.sqrt(SE/n_tests)
RMSE = sqrt(((pred_arr-real_rate_arr)**2).sum()/n_tests)
print('User-user CF, RMSE =', RMSE)
# Xét trên ngưỡng 4.5 là thích - value = true


def F1_threshold(pred_arr, real_rate_arr, t):
    pred_arr = pred_arr >= t
    real_rate_arr = real_rate_arr >= t
    pred_arr = pred_arr.astype(int)
    real_rate_arr = real_rate_arr.astype(int)
    F1 = f1_score(real_rate_arr, pred_arr.round(),
                  average='binary', zero_division=1)
    return F1


# trung bình đánh giá tất cả user (global)
i_arr = rate_test[:, 1]
mean_i = rs.mean_rating_item_i(i_arr)
ratings = rs.Y_data[:, 2]
print(ratings)
m = np.mean(ratings)
print(m)
# trung bình thang đánh giá, global, local
t = (thres_arr+m+3)/3
#t = (2*thres_arr+m+3)/4

print("F1_threshold >=thres_arr")
print(F1_threshold(pred_arr, real_rate_arr, t))

'''pred_arr = pred_arr >= 4.5
real_rate_arr = real_rate_arr >= 4.5
pred_arr = pred_arr.astype(int)
real_rate_arr = real_rate_arr.astype(int)
F1 = f1_score(real_rate_arr, pred_arr.round(), labels=[0, 1],
              average='macro', zero_division=1)
print("F1:")
print(F1)'''
# F1 = precision_recall_fscore_support( real_rate_arr, pred_arr.round(), average = 'macro', zero_division = 1)
'''print("f1_score nè:", F1)
rc = recall_score(real_rate_arr, pred_arr.round(), average="macro")
print("recall_score nè:", rc)
ps = precision_score(real_rate_arr, pred_arr.round(), average="macro")
print("precision_score nè:", ps)'''
'''print("rate_train test: ", rate_train)
print("rate_train.shape", rate_train.shape)
print("rate test: ", rate_test)
print("rate_test.shape", rate_test.shape)
'''
