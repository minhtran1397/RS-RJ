import pandas as pd
import numpy as np  # xử lý mảng
from scipy import sparse  # spare: thưa thớt, chuẩn hóa ma trận
# thư viện thuật toán (có độ tương đồng với gì nữa á)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
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
from scipy.spatial.distance import jaccard
from scipy.spatial.distance import correlation
# from scipy.spatial import distance


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
        print('Ybar_data nè:')
        print(self.Ybar_data)
        # tạo mảng 0 với số lương phần tử = n_user (số lượng dòng)
        self.mu = np.zeros((self.n_users,))
        print('mu nè:')
        print(self.mu)
        for n in range(self.n_users):
            # mỗi dòng là chỉ số đánh giá của mỗi người dùng
            # chỉ số đánh giá phải là int nên cần convert
            print('n nè:')
            print(n)
            ids = np.where(users == n)[0].astype(np.int32)
            print('ids nè:')
            print(ids)
            # chỉ ra các item liên quan đến user ids, dòng thứ ids, cột 1
            item_ids = self.Y_data[ids, 1]
            print('item_ids nè:')
            print(item_ids)
            # chỉ ra các rating liên quan đến user ids, dòng thứ ids, cột 2
            ratings = self.Y_data[ids, 2]
            print('ratings nè:')
            print(ratings)
            # tính trung bình đánh giá matrix
            m = np.mean(ratings)
            print('trung bình đánh giá m nè:')
            print(m)
            if np.isnan(m):  # Nếu giá trị m rỗng hoặc không phải số thì m=0
                m = 0  # to avoid empty array and nan value
            # normalize: FIXME: chuẩn hóa rating sang rating trừ trung bình ?!!!
            self.Ybar_data[ids, 2] = ratings - \
                self.mu[n]  # FIXME: Không chạy ?
            print('Ybar_data chuẩn hóa nè:')
            print(self.Ybar_data)

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
        print('Ybar1 - mảng i-u-r chưa sắp xếp:')
        print(self.Ybar)
        self.Ybar = self.Ybar.tocsr()  # sắp xếp lại theo thứ tự tăng dần id item
        print('Ybar1 - mảng i-u-r đã sắp xếp id item tăng dần:')
        print(self.Ybar)
        print('Ybar2.T - mảng i-u-r chuyển vị thành mảng u-i-r:')
        print(self.Ybar.T)
        # TODO: Tìm
    # Chuyển ma trận thành 0 1 (binary) để tính Jaccard

    """def intArrToBinary(self):
        self.BinArr = np.where(self.Ybar.T > 0, 1, 0)"""

    # tính độ tương đồng

    def similarity(self, method=1):
        if method == 1:  # cosine
            # TODO: tìm .T là gì: .T là ma trận nghịch đảo, ở đây nghịch đảo vị trí user-item thành vị trí item-user
            self.S = self.dist_func(self.Ybar.T)
        # jaccard np.where(a > 0.5, 1, 0) ý tưởng là dùng hàm trên id của item non zero
        if method == 2:
            y_true = np.array([[0, 1, 1], [0, 1, 0]])
            y_pred = np.array([[1, 1, 1], [1, 0, 0]])
            self.S = self.dist_func(y_true[0], y_pred[0])
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
