import pandas as pd
import numpy as np  # xử lý mảng
from scipy import sparse  # spare: thưa thớt, chuẩn hóa ma trận
# thư viện thuật toán (có độ tương đồng với gì nữa á)
from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.metrics import jaccard_score
'''
From scikit - learn: [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’].
These metrics support sparse matrix inputs.
[‘nan_euclidean’] but it does not yet support sparse matrices.

From scipy.spatial.distance: [‘braycurtis’, ‘canberra’, ‘chebyshev’,
‘correlation’, ‘dice’, ‘hamming’, ‘jaccard’, ‘kulsinski’,
‘mahalanobis’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’,
‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’,
‘yule’] See the documentation for scipy.spatial.distance for details on these metrics.
These metrics do not support sparse matrix inputs.
'''
# from scipy.spatial.distance import jaccard
from scipy.spatial.distance import correlation
# from scipy.spatial import distance


def jaccard_similarity(x, y):
    """ returns the jaccard similarity between two lists """
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality / float(union_cardinality)


def rjaccard_similarity(u, v):
    intersection_cardinality = len(set.intersection(*[set(u), set(v)]))
    Iu = len(set(u))
    Iv = len(set(v))
    Iu_bar = Iu - intersection_cardinality
    Iv_bar = Iv - intersection_cardinality
    if intersection_cardinality == 0:
        return 0
    else:
        return(1/(1+(1/intersection_cardinality)+(Iu_bar/(1+Iu_bar))+(1/(1+Iv_bar))))


def common_dimensions(u, v):
    """ takes in two vectors and returns a tuple of the vectors with both non zero dimensions
       i.e.
       v1 : [ 1, 2, 3, 0 ]        -->      return [2, 3]
       v2 : [ 0, 4, 5, 6 ]        -->      return [4, 5]  """
    common_u = []
    common_v = []
    for i in range(0, len(u)):
        if u[i] != 0 and v[i] != 0:
            common_u.append(u[i])
            common_v.append(v[i])
            # print 'INDEX SAME:',i
    return common_u, common_v


def msd_similarity(u, v, l):
    common_arr = common_dimensions(u, v)
    msd = sum(pow(a - b, 2) for a, b in zip(common_arr[0], common_arr[1]))
    intersection_cardinality = len(common_arr[0])
    return (1-((msd/intersection_cardinality)/l))


class CF(object):
    """docstring for CF"""

#    def __init__(self, Y_data, k, dist_func=cosine_similarity, uuCF=1):
    def __init__(self, Y_data, k, dist_func, uuCF=1):
        self.uuCF = uuCF  # user-user (1) or item-item (0) CF
        # TODO: Xem lại Y_data: dữ liệu đầu vào
        self.Y_data = Y_data if uuCF else Y_data[:, [1, 0, 2]]
        # Theo như Code: Nếu là u-u, thì là user-item-rating, còn nếu xét i-i thì đảo cột 0 sang cột 1
        self.k = k  # number of neighbor points
        self.dist_func = dist_func  # hàm tính độ tương đồng (ở đây là COS)
        self.Ybar_data = None  # TODO: Xem lại Ybar_data có phải ma trận user-item không
        # number of users and items. Remember to add 1 since id starts from 0
        self.n_users = int(np.max(self.Y_data[:, 0])) + 1
        self.n_items = int(np.max(self.Y_data[:, 1])) + 1
        self.L = int(max(self.Y_data[:, 2]))

    '''
    Khi có dữ liệu mới, cập nhận Utility matrix
    bằng cách thêm các hàng này vào cuối Utility Matrix.
    # TODO: Để cho đơn giản, giả sử rằng không có users hay items mới,
    cũng không có ratings nào bị thay đổi.
    '''

    def add(self, new_data):
        """
            Update Y_data matrix when new ratings come.
            For simplicity, suppose that there is no new user or item.
            """
        self.Y_data = np.concatenate((self.Y_data, new_data), axis=0)

        # axis = 0: thêm dòng mới vào ma trận
        # axis = 1: thêm cột mới vào ma trận
    # Tính toán normalized utility matrix và Similarity matrix

    def normalize_Y(self):
        users = self.Y_data[:, 0]  # all users - first col of the Y_data
        self.Ybar_data = self.Y_data.copy()  # copy Y_data qua Ybar_data
        # tạo mảng 0 với số lương phần tử = n_user (số lượng dòng)
        self.mu = np.zeros((self.n_users,))
        for n in range(self.n_users):
            # mỗi dòng là chỉ số đánh giá của mỗi người dùng
            # chỉ số đánh giá phải là int nên cần convert
            ids = np.where(users == n)[0].astype(np.int32)
            # chỉ ra các item liên quan đến user ids, dòng thứ ids, cột 1
            item_ids = self.Y_data[ids, 1]
            # chỉ ra các rating liên quan đến user ids, dòng thứ ids, cột 2
            ratings = self.Y_data[ids, 2]
            # tính trung bình đánh giá matrix
            m = np.mean(ratings)
            if np.isnan(m):  # Nếu giá trị m rỗng hoặc không phải số thì m=0
                m = 0  # to avoid empty array and nan value
            # normalize: FIXME: chuẩn hóa rating sang rating trừ trung bình ?!!!
            self.Ybar_data[ids, 2] = ratings - \
                self.mu[n]  # FIXME: Không chạy ?

        ################################################
        # sparse matrix là ma trận hiểu số 0 là rỗng, chỉ lưu trữ vị trí của nó
        # form the rating matrix as a sparse matrix. Sparsity is important
        # for both memory and computing efficiency. For example, if #user = 1M,
        # #item = 100k, then shape of the rating matrix would be (100k, 1M),
        # you may not have enough memory to store this. Then, instead, we store
        # nonzeros only, and, of course, their locations.
        # Tạo mảng ((item, user) rating)
        self.Ybar = sparse.coo_matrix((self.Ybar_data[:, 2],
                                       (self.Ybar_data[:, 1], self.Ybar_data[:, 0])), (self.n_items, self.n_users))
        self.Ybar = self.Ybar.tocsr()  # sắp xếp lại theo thứ tự tăng dần id item
        print('Ybar')
        print(self.Ybar)
        # TODO: Tìm
    # Chuyển ma trận thành 0 1 (binary) để tính Jaccard

    """def intArrToBinary(self):
        self.BinArr = np.where(self.Ybar.T > 0, 1, 0)"""

    # tính độ tương đồng

    def similarity(self, method=1):
        if method == 1:  # cosine
            # TODO: tìm .T là gì: .T là ma trận nghịch đảo, ở đây nghịch đảo vị trí user-item thành vị trí item-user
            self.S = self.dist_func(self.Ybar.T, self.Ybar.T)
        # jaccard np.where(a > 0.5, 1, 0) ý tưởng là dùng hàm trên id của item non zero
        if method == 2:
            self.sim_jaccard = np.zeros((self.n_users, self.n_users))
            users = self.Y_data[:, 0]
            print(self.sim_jaccard[0, 1])
            for i in range(self.n_users):
                ids = np.where(users == i)[0].astype(np.int32)
                item_ids = self.Y_data[ids, 1]
                print('u{}={}'.format(i, item_ids))
                # self.sim_jaccard[ids, ids] = 1
                for j in range(self.n_users):
                    jds = np.where(users == j)[0].astype(np.int32)
                    item_jds = self.Y_data[jds, 1]
                    print('v{}={}'.format(j, item_jds))
                    print("-----")
                    self.sim_jaccard[i, j] = self.dist_func(
                        item_ids, item_jds)
                self.sim_jaccard[i]
            self.S = self.sim_jaccard
        if method = 3:  # msd_similarity
            # TODO:
        print('S nè :')
        print(self.S)
    # Thực hiện lại 2 hàm phía trên khi có thêm dữ liệu.

    def refresh(self, method=1):
        """
        Normalize data and calculate similarity matrix again (after
        some few ratings added)
        """
        self.normalize_Y()
        self.similarity(method)

    def fit(self, method=1):
        self.refresh(method)
