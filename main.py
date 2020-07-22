from classCF import *
# data file user-user CF:
r_cols = ['user_id', 'item_id', 'rating']
ratings = pd.read_csv('ex.dat', sep=' ', names=r_cols, encoding='latin-1')
Y_data = ratings.to_numpy()  # as_matrix() không chạy
COS = cosine_similarity  # method = 1
JAC = jaccard_score  # method = 2
#rs = CF(Y_data, k=2, dist_func=COS, uuCF=1)
rs = CF(Y_data, k=2, dist_func=JAC, uuCF=1)
# rs.fit()
print("############ Bắt đầu đầu đầu ####################")
rs.fit(2)
# rs.normalize_Y()
# print(rs.similarity())
print("xxxxxxxxxxxx Kết thúc xxxxxxxxxxxxxxxxxxxx")
