from classCF import *
# data file user-user CF:
r_cols = ['user_id', 'item_id', 'rating']
ratings = pd.read_csv('ex.dat', sep=' ', names=r_cols, encoding='latin-1')
Y_data = ratings.to_numpy()  # as_matrix() không chạy
COS = cos_similarity  # method = 1
JAC = rjaccard_similarity  # method = 2
MSD = msd_similarity  # method = 3
COR = cor_similarity  # method = 4
CPC = cpc_similarity  # method = 5

#rs = CF(Y_data, k=2, dist_func=COS, uuCF=1)

rs = CF(Y_data, k=2, dist_func=CPC, uuCF=1)
# rs.fit()
print("############ Bắt đầu đầu đầu ####################")
# fit(method)
rs.fit(5)
# rs.normalize_Y()
# print(rs.similarity())
print("xxxxxxxxxxxx Kết thúc xxxxxxxxxxxxxxxxxxxx")
