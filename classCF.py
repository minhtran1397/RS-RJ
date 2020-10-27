import pandas as pd
import numpy as np  # xử lý mảng
from scipy import sparse  # spare: thưa thớt, chuẩn hóa ma trận
import math
from math import *
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
# from scipy.spatial.distance import correlation
# from scipy.spatial import distance


def urp_similarity(u, v):
    avg_u = np.mean(u.data)
    avg_v = np.mean(v.data)
    var_u = sqrt(((u.data - avg_u) ** 2).sum() / len(u.data))
    var_v = sqrt(((v.data - avg_v) ** 2).sum() / len(v.data))
    return (1-(1/(1+exp(-abs(avg_u-avg_v)*(abs(var_u-var_v))))))


def jaccard_similarity(x, y):
    """ returns the jaccard similarity between two lists """
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality / float(union_cardinality)


def sdc_similarity(x, y):
    """ returns the jaccard similarity between two lists """
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    return (2*intersection_cardinality) / (x.size+y.size)


def sci_similarity(x, y):
    """ returns the jaccard similarity between two lists """
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    return (intersection_cardinality) / sqrt(x.size*y.size)


def overlap_similarity(x, y):
    """ returns the jaccard similarity between two lists """
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    return (intersection_cardinality) / min(x.size, y.size)


def triangle_similarity(common, u, v):
    common_u = u[:, common].toarray()[0]
    common_v = v[:, common].toarray()[0]
    num = sqrt(((common_u-common_v)**2).sum())
    den = sqrt((common_u ** 2).sum()) + sqrt((common_v ** 2).sum())
    if common.size == 0:
        return 0
    else:
        return (1 - (num / den))


def tmj_similarity(common, ids, jds, u, v):
    jac = jaccard_similarity(ids, jds)
    triangle = triangle_similarity(common, u, v)
    return jac*triangle


def ctj_similarity(common, ids, jds, u, v):
    jac = jaccard_similarity(ids, jds)
    triangle = triangle_similarity(common, u, v)
    return 0.5*jac*(triangle+1)


def my_jaccard_similarity(common, ids, jds, u, v):
    common_u = u[:, common].toarray()[0]
    common_v = v[:, common].toarray()[0]
    print('common_u', common_u)
    print('common_v', common_v)
    num = 0
    for i in range(common_u.size):
        print('common_u[i]', common_u[i])
        print('common_v[i]', common_v[i])
        if (common_u[i] >= 3 and common_v[i] >= 3):
            num += 1
        elif (common_u[i] < 3 and common_v[i] < 3):
            num += 1
    union_cardinality = len(set.union(*[set(ids), set(jds)]))
    return num / float(union_cardinality)


def jsmcc_similarity(common, ids, jds, u, v):
    jac = jaccard_similarity(ids, jds)
    intersection_cardinality = len(set.intersection(*[set(ids), set(jds)]))
    common_u = u[:, common].toarray()[0]
    common_v = v[:, common].toarray()[0]
    num = 0
    for i in range(common_u.size):
        if (common_u[i] > 3 and common_v[i] > 3):
            num += 1
        elif (common_u[i] < 3 and common_v[i] < 3):
            num += 1
        elif (common_u[i] == 3 and common_v[i] == 4):
            num += 1
        elif (common_u[i] == 3 and common_v[i] == 2):
            num += 1
        elif (common_u[i] == 3 and common_v[i] == 3):
            num += 1
    if intersection_cardinality == 0:
        return 0
    else:
        return (jac*(num/intersection_cardinality))


def uod_similarity(common, u, v):
    # số lượng common
    m = common.size
    rmax_u = np.max(u.data)
    rmin_u = np.min(u.data)
    common_u = u[:, common].toarray()[0]
    common_v = v[:, common].toarray()[0]
    if m == 0:
        return 0
    elif (sum(abs(common_u - common_v)) == 0):
        return (sqrt(m*((rmax_u - rmin_u)**2))/0.9)
    else:
        return (sqrt(m*((rmax_u - rmin_u)**2))/(sqrt(sum((common_u - common_v)**2))))


def jacuod_similarity(common, ids, jds, u, v):
    jac = jaccard_similarity(ids, jds)
    uod = uod_similarity(common, u, v)
    return jac * uod


def pip_similarity1(common, ids, jds, u, v, mean_common):
    r_med = 3  # trung binh thanh danh gia
    r_u = u[:, common].toarray()[0]
    r_v = v[:, common].toarray()[0]
    pip = 0
    for i in range(r_u.size):
        if (r_u[i] > r_med and r_v[i] < r_med) or (r_u[i] < r_med and r_v[i] > r_med):
            proximity = pow(((2 * 4 + 1) - abs(r_u[i] - r_v[i])), 2)
            impact = (abs(r_u[i] - r_med) + 1) * (abs(r_v[i] - r_med) + 1)
        else:
            proximity = pow(((2 * 4 + 1) - (2*abs(r_u[i] - r_v[i]))), 2)
            impact = 1/((abs(r_u[i] - r_med) + 1) * (abs(r_v[i] - r_med) + 1))
        if (r_u[i] > mean_common[i] and r_v[i] > mean_common[i]) or (r_u[i] < mean_common[i] and r_v[i] < mean_common[i]):
            popularity = 1 + pow((((r_u[i] + r_v[i]) / 2) - mean_common[i]), 2)
        else:
            popularity = 1
        pip += (proximity * impact * popularity)
    return pip


def pip_similarity(common, ids, jds, u, v, mean_common):
    r_med = 0.5  # trung binh thanh danh gia
    common_u = u[:, common].toarray()[0]
    common_v = v[:, common].toarray()[0]
    # Chuyển giá trị đánh giá sang thang điểm [0,1]
    # [1,2,3,4,5] -> [0,0.25,0.5,0.75,1]
    mean_common = np.where(mean_common, (1/4)*(mean_common-1), mean_common)
    r_u = np.where(common_u, (1/4)*(common_u-1), common_u)
    r_v = np.where(common_v, (1/4)*(common_v-1), common_v)
    pip = 0
    for i in range(r_u.size):
        if (r_u[i] > r_med and r_v[i] < r_med) or (r_u[i] < r_med and r_v[i] > r_med):
            proximity = pow(((2 * 4 + 1) - (2*abs(r_u[i] - r_v[i]))), 2)
            impact = 1/((abs(r_u[i] - r_med) + 1) *
                        (abs(r_v[i] - r_med) + 1))
        else:
            proximity = pow(((2 * 4 + 1) - abs(r_u[i] - r_v[i])), 2)
            impact = (abs(r_u[i] - r_med) + 1) * \
                (abs(r_v[i] - r_med) + 1)
        if (r_u[i] > mean_common[i] and r_v[i] > mean_common[i]) or (r_u[i] < mean_common[i] and r_v[i] < mean_common[i]):
            popularity = 1 + pow((((r_u[i] + r_v[i]) / 2) - mean_common[i]), 2)
        else:
            popularity = 1
        pip += (proximity * impact * popularity)
    return pip


def nhsm_similarity(common, ids, jds, u, v, mean_common):
    intersection_cardinality = len(set.intersection(*[set(ids), set(jds)]))
    jac_ = intersection_cardinality/(ids.size*jds.size)
    pss = pss_similarity(common, ids, jds, u, v, mean_common)
    urp = urp_similarity(u, v)
    return jac_*pss*urp


def jpss_similarity(common, ids, jds, u, v, mean_common):
    intersection_cardinality = len(set.intersection(*[set(ids), set(jds)]))
    jac_ = intersection_cardinality / (ids.size * jds.size)
    print('jac_', jac_)
    pss = pss_similarity(common, ids, jds, u, v, mean_common)
    print('pss', pss)

    return jac_*pss


def pss_similarity(common, isd, jds, u, v, mean_common):
    r_med = 3  # trung binh thanh danh gia
    r_u = u[:, common].toarray()[0]
    r_v = v[:, common].toarray()[0]
    pss = 0
    for i in range(r_u.size):
        proximity = 1 - (1 / (1 + exp(-abs(r_u[i] - r_v[i]))))
        significance = 1 / \
            (1 + exp(-abs(r_u[i] - r_med) * abs(r_v[i] - r_med)))
        singularity = 1-(1/(1+exp(-abs((r_u[i]+r_v[i])/2-mean_common[i]))))
        pss += (proximity * significance * singularity)
    return pss


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


def jaclmh_similarity(common, u, v):
    ul_1 = u >= 1
    ul_2 = u <= 2
    ul = np.intersect1d(ul_1.indices, ul_2.indices)
    um_2 = u > 2
    um_5 = u < 5
    um = np.intersect1d(um_2.indices, um_5.indices)
    uh_5 = u >= 5
    uh = uh_5.indices
    '''print("ul", ul)
    print("um", um)
    print("uh", uh)'''

    vl_1 = v >= 1
    vl_2 = v <= 2
    vl = np.intersect1d(vl_1.indices, vl_2.indices)
    vm_2 = v > 2
    vm_5 = v < 5
    vm = np.intersect1d(vm_2.indices, vm_5.indices)
    vh_5 = v >= 5
    vh = vh_5.indices
    '''print("vl", vl)
    print("vm", vm)
    print("vh", vh)'''
    jacl = 0
    jacm = 0
    jach = 0
    if ul.size == 0 and vl.size == 0:
        jacl = 0
    else:
        jacl += len(set.intersection(*[set(ul), set(vl)])
                    ) / len(set.union(*[set(ul), set(vl)]))
    if um.size == 0 and vm.size == 0:
        jacm = 0
    else:
        jacm += len(set.intersection(*[set(um), set(vm)])
                    ) / len(set.union(*[set(um), set(vm)]))
    if uh.size == 0 and vh.size == 0:
        jach = 0
    else:
        jach += len(set.intersection(*[set(uh), set(vh)])
                    ) / len(set.union(*[set(uh), set(vh)]))
    return (jacl+jacm+jach)/3


def jlmhuod_similarity(common, u, v):
    uod = uod_similarity(common, u, v)
    jlmh = jaclmh_similarity(common, u, v)
    return uod*jlmh


def common_dimensions(u, v):
    '''takes in two vectors and returns a tuple of the vectors with both non zero dimensions
        i.e.
        v1 : [ 1, 2, 3, 0 ]        -->      return [2, 3]
        v2 : [ 0, 4, 5, 6 ]        -->      return [4, 5] '''
    common_u = []
    common_v = []
    for i in range(len(u)):
        if u[i] != 0 and v[i] != 0:
            common_u.append(u[i])
            common_v.append(v[i])
            # print 'INDEX SAME:',i
    return common_u, common_v


def common_dimensions1(common, u, v):
    '''takes in two vectors and returns a tuple of the vectors with both non zero dimensions
        i.e.
        v1 : [ 1, 2, 3, 0 ]        -->      return [2, 3]
        v2 : [ 0, 4, 5, 6 ]        -->      return [4, 5] '''
    common_u = u[:, common].toarray()
    common_v = v[:, common].toarray()
    # common = np.intersect1d(item_ids, item_jds)
    # common_u = []
    # common_v = []
    # print("common", common)
    # for i in range(len(item_ids)):
    # print("v[:, common].data", v[:, common].data)
    # print("u[:, common].data", u[:, common].data)
    # for i in range(len(common)):
    # for i in range(item_ids.shape[]):
    # print("v[item_ids[i]]", v[:, item_ids[i]].data[0])
    # if v[item_ids[i]] != 0:
    # if v[:, item_ids[i]].data
    # common_v = np.append(common_v, v[:, common].data, axis=0)
    # common_u = np.append(common_u, u[:, common].data, axis=0)
    # print("common_u", common_u)
    # common_v.append(v[:, item_ids[i]].data[0])
    # common_u.append(u[:, item_ids[i]].data[0])
    # print 'INDEX SAME:',i
    # print("common_u", common_u)
    # print("common_v", common_v)
    return common_u[0], common_v[0]


def msd_similarity(common, u, v):
    l = 16
    # common_arr = common_dimensions1(item_ids, u, v)
    common_u = u[:, common].toarray()[0]
    common_v = v[:, common].toarray()[0]
    msd = ((common_u-common_v)**2).sum()
    intersection_cardinality = common.size
    if intersection_cardinality == 0:
        return 0
    else:
        return (1-((msd/intersection_cardinality)/l))


def msd_similarity_pp(common, u, v):
    # common_arr = common_dimensions1(item_ids, u, v)
    common_u = u[:, common].toarray()[0]
    common_v = v[:, common].toarray()[0]
    # Chuyển giá trị đánh giá sang thang điểm [0,1]
    # [1,2,3,4,5] -> [0,0.25,0.5,0.75,1]
    common_u = np.where(common_u, (1/4)*(common_u-1), common_u)
    common_v = np.where(common_v, (1/4)*(common_v-1), common_v)
    msd = sum(pow(a - b, 2) for a, b in zip(common_u, common_v))
    intersection_cardinality = len(common_u)
    if intersection_cardinality == 0:
        return 0
    else:
        return (1-(msd/intersection_cardinality))


def amsd_similarity(common, ids, jds, u, v):
    msd = msd_similarity(common, u, v)
    intersection_cardinality = len(set.intersection(*[set(ids), set(jds)]))
    Iu = len(set(ids))
    Iv = len(set(jds))
    sorensen_index = ((2*intersection_cardinality)/(Iu+Iv))
    aus = intersection_cardinality / Iu
    return msd*sorensen_index*aus


def srj_similarity(ids, jds, u, v):
    rj = rjaccard_similarity(ids, jds)
    intersection_cardinality = len(set.intersection(*[set(ids), set(jds)]))
    Iu = len(set(ids))
    Iv = len(set(jds))
    sorensen_index = ((2*intersection_cardinality)/(Iu+Iv))
    return rj*sorensen_index


def square_rooted(x, y=0):
    """ return 3 rounded square rooted value """
    # return round(sqrt(sum([(a - y) * (a - y) for a in x])), 3)
    return sqrt(sum([(a - y) * (a - y) for a in x]))


def jmsd_similarity(common, ids, jds, u, v):
    j = jaccard_similarity(ids, jds)
    msd = msd_similarity(common, u, v)
    return j*msd


def rjmsd_similarity(common, ids, jds, u, v):
    rj = rjaccard_similarity(ids, jds)
    msd = msd_similarity(common, u, v)
    return rj*msd


def rjurp_similarity(ids, jds, u, v):
    rj = rjaccard_similarity(ids, jds)
    urp = urp_similarity(u, v)
    return rj * urp


def rpb_similarity(common, u, v):
    avg_u = np.mean(u.data)
    avg_v = np.mean(v.data)
    var_u = sqrt(sum([(i - avg_u) ** 2 for i in u.data]) / u.size)
    var_v = sqrt(sum([(j - avg_v) ** 2 for j in v.data]) / v.size)
    s = abs(avg_u - avg_v) * abs(var_u - var_v)
    return cos(s)


def rating_jaccard_rpb_similarity(common, u, v):
    rating_jacard = rating_jaccard_similarity(common, u, v)
    rpb = rpb_similarity(common, u, v)
    print("rating_jacard", rating_jacard)
    print("rpb", rpb)
    return rating_jacard*rpb


def rj_rpb_similarity(common, ids, jds, u, v):
    rj = rjaccard_similarity(ids, jds)
    rpb = rpb_similarity(common, u, v)
    return rj*rpb


def rating_jaccard_similarity(common, u, v):
    ru = u[:, common].toarray()[0]
    rv = v[:, common].toarray()[0]
    # print('common.size', common.size)
    nt = 0
    if common.size == 0:
        return 0
    elif (abs(np.mean(u.data) - np.mean(v.data)) < 0.3):
        return 1
    else:
        for i in range(common.size):
            if (ru[i] == rv[i]):  # or
                nt += 1
        return nt / common.size


def cos_similarity(common, u, v):
    # common_arr = common_dimensions1(common, u, v)
    # print("common_arr", common_arr)
    '''common_u = common_arr[0]
    common_v = common_arr[1] '''
    common_u = u[:, common].toarray()[0]
    common_v = v[:, common].toarray()[0]
    # numerator = sum(a * b for a, b in zip(common_u, common_v))
    numerator = (common_u * common_v).sum()
    denominator = square_rooted(
        common_u)*square_rooted(common_v)
    if denominator == 0:
        return 0
    else:
        return numerator / float(denominator)


"""def average(my_list):
    if len(my_list) == 0:
        return 0
    else:
        avg = float(sum(my_list)) / float(len(my_list))
        return avg"""


def nonzero_count(my_list):
    """
    takes in a list an returns the amount of non zero values in the list
    i.e.
    list [0 0 0 0 1 2 3 ] --> returns 3
    """
    counter = 0
    for value in my_list:
        if value != 0:
            counter += 1
    return counter


def cor_similarity(common, u, v):
    # rm: lấy average(common):
    '''common_arr = common_dimensions1(common, u, v)
    common_u = common_arr[0]
    common_v = common_arr[1] '''
    avg_u = np.mean(u.data)
    avg_v = np.mean(v.data)
    common_u = u[:, common].toarray()[0]-avg_u
    common_v = v[:, common].toarray()[0] - avg_v
    # avg_u = sum(u) / nonzero_count(u)
    # print("sum(u): ", sum(u))
    # print("sum(v): ", sum(v))
    # avg_v = sum(v) / nonzero_count(v)
    # print("avg_u: ", avg_u)
    # print("avg_v: ", avg_v)
    # numerator = sum((a - avg_u)*(b - avg_v)for a, b in zip(common_u, common_v))
    numerator = (common_u * common_v).sum()
    denominator = sqrt(((common_u)**2).sum())*sqrt(((common_v)**2).sum())
    if denominator == 0:
        return 0
    else:
        # return round(numerator / float(denominator), 3)
        return numerator / float(denominator)


def WPC_similarity(common, numofratingperitem, u, v):
    # rm: lấy average(common):
    '''common_arr = common_dimensions1(common, u, v)
    common_u = common_arr[0]
    common_v = common_arr[1] '''
    avg_u = np.mean(u.data)
    avg_v = np.mean(v.data)
    common_u = u[:, common].toarray()[0]-avg_u
    common_v = v[:, common].toarray()[0] - avg_v
    mj = numofratingperitem[common]
    W = np.log(mj/924)
    # avg_u = sum(u) / nonzero_count(u)
    # print("sum(u): ", sum(u))
    # print("sum(v): ", sum(v))
    # avg_v = sum(v) / nonzero_count(v)
    # print("avg_u: ", avg_u)
    # print("avg_v: ", avg_v)
    # numerator = sum((a - avg_u)*(b - avg_v)for a, b in zip(common_u, common_v))
    numerator = (common_u * common_v).sum()
    denominator = square_rooted(common_u)*square_rooted(common_v)
    if denominator == 0:
        return 0
    else:
        # return round(numerator / float(denominator), 3)
        return numerator / float(denominator)


def cpc_similarity(common, u, v):
    # rm: lấy average(common):
    rm = 3
    '''common_arr = common_dimensions1(common, u, v)
    common_u = common_arr[0]
    common_v = common_arr[1] '''
    common_u = u[:, common].toarray()[0]
    common_v = v[:, common].toarray()[0]
    numerator = sum((a - rm)*(b - rm)
                    for a, b in zip(common_u, common_v))
    denominator = square_rooted(
        common_u, rm)*square_rooted(common_v, rm)
    if denominator == 0:
        return 0
    else:
        return round(numerator / float(denominator), 3)


class CF(object):
    """docstring for CF"""

#    def __init__(self, Y_data, k, dist_func=cosine_similarity, uuCF=1):
    def __init__(self, Y_data, k, dist_func=cosine_similarity, uuCF=1):
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

    def normalize_Yzzzzzzz(self):
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
            # print("m nè: ", m)
            if np.isnan(m):  # Nếu giá trị m rỗng hoặc không phải số thì m=0
                m = 0  # to avoid empty array and nan value
            self.mu[n] = m
            # normalize: FIXME: chuẩn hóa rating sang rating trừ trung bình ?!!!
            self.Ybar_data[ids, 2] = ratings  # - self.mu[n]
            # print("mu nè: ", self.mu[n])
        ################################################
        # sparse matrix là ma trận hiểu số 0 là rỗng, chỉ lưu trữ vị trí của nó
        # form the rating matrix as a sparse matrix. Sparsity is important
        # for both memory and computing efficiency. For example, if #user = 1M,
        # #item = 100k, then shape of the rating matrix would be (100k, 1M),
        # you may not have enough memory to store this. Then, instead, we store
        # nonzeros only, and, of course, their locations.
        # Tạo mảng ((item, user) rating)
        # self.Ybar = sparse.coo_matrix((self.Ybar_data[:, 2],(self.Ybar_data[:, 1], self.Ybar_data[:, 0])), (self.n_items, self.n_users))'''
        # Tạo mảng ((item, user) rating)
        self.Ybar = sparse.coo_matrix((self.Ybar_data[:, 2],
                                       (self.Ybar_data[:, 0], self.Ybar_data[:, 1])), (self.n_users, self.n_items))
        self.Ybar = self.Ybar.tocsr()  # sắp xếp lại theo thứ tự tăng dần id item
        # self.Ybar = self.Ybar.toarray()
        print('users-items')
        print(self.Ybar)

    def normalize_Y(self):
        users = self.Y_data[:, 0]  # all users - first col of the Y_data
        self.Ybar_data = self.Y_data.copy()  # copy Y_data qua Ybar_data
        # tạo mảng 0 với số lương phần tử = n_user (số lượng dòng)
        self.mu = np.zeros((self.n_users,))
        for n in range(self.n_users):
            # mỗi dòng là chỉ số đánh giá của mỗi người dùng
            # chỉ số đánh giá phải là int nên cần convert
            ids = np.where(users == n)[0]  # .astype(np.int32)
            # chỉ ra các item liên quan đến user ids, dòng thứ ids, cột 1
            item_ids = self.Y_data[ids, 1]
            # chỉ ra các rating liên quan đến user ids, dòng thứ ids, cột 2
            ratings = self.Y_data[ids, 2]
            # tính trung bình đánh giá matrix
            m = np.mean(ratings)
            # print("m nè: ", m)
            if np.isnan(m):  # Nếu giá trị m rỗng hoặc không phải số thì m=0
                m = 0  # to avoid empty array and nan value
            self.mu[n] = m
            # normalize: FIXME: chuẩn hóa rating sang rating trừ trung bình ?!!!
            # print("ratings - self.mu[n]", ratings - self.mu[n])
            self.Ybar_data[ids, 2] = ratings-self.mu[n]

        ################################################
        # sparse matrix là ma trận hiểu số 0 là rỗng, chỉ lưu trữ vị trí của nó
        # form the rating matrix as a sparse matrix. Sparsity is important
        # for both memory and computing efficiency. For example, if #user = 1M,
        # #item = 100k, then shape of the rating matrix would be (100k, 1M),
        # you may not have enough memory to store this. Then, instead, we store
        # nonzeros only, and, of course, their locations.
        self.Ybar = sparse.coo_matrix((self.Ybar_data[:, 2],
                                       (self.Ybar_data[:, 0], self.Ybar_data[:, 1])), (self.n_users, self.n_items))
        # self.Ybar = self.Ybar.tocsr()  # sắp xếp lại theo thứ tự tăng dần id item
        self.Ybar = self.Ybar.toarray()
        print('users-items')
        print(self.Ybar)
        '''# Tạo mảng ((item, user) rating)
        self.Ybar = sparse.coo_matrix((self.Ybar_data[:, 2],
                                       (self.Ybar_data[:, 1], self.Ybar_data[:, 0])), (self.n_items, self.n_users))
        # Tạo mảng ((item, user) rating)
        self.Y = sparse.coo_matrix((self.Ybar_data[:, 2],
                                    (self.Ybar_data[:, 1], self.Ybar_data[:, 0])), (self.n_items, self.n_users))
        self.Y = self.Y.tocsr()
        print('Y.T')
        print(self.Y.T)
        self.Ybar = sparse.coo_matrix((self.Y_data[:, 2],
                                       (self.Y_data[:, 1], self.Y_data[:, 0])), (self.n_items, self.n_users))
        self.Ybar = self.Ybar.tocsr()  # sắp xếp lại theo thứ tự tăng dần id item
        print('Ybar.T')
        print(self.Ybar.T)'''
    # Chuyển ma trận thành 0 1 (binary) để tính Jaccard

    """def intArrToBinary(self):
        self.BinArr = np.where(self.Ybar.T > 0, 1, 0)"""
    # Lặp qua mỗi cặp user để tính độ tương đồng (dựa trên mảng coo_matrix Ybar)

    def loop_user(self):
        self.sim_arr = np.zeros((self.n_users, self.n_users))
        # print(self.n_users)
        for i in range(self.n_users):
            # ids = np.where(users == i)[0].astype(np.int32)
            ratings_i_nD = self.Ybar.T[i, :].toarray()
            # Xuất ra mảng nhiều chiều [[value]], cần truyền mảng 1 chiều [value] => :
            ratings_i = ratings_i_nD[0]
            print('u{}={}'.format(i, ratings_i))
            for j in range(self.n_users):
                # jds = np.where(users == j)[0].astype(np.int32)
                ratings_j_nD = self.Ybar.T[j, :].toarray()
                # Xuất ra mảng nhiều chiều [[value]], cần truyền mảng 1 chiều [value] => :
                ratings_j = ratings_j_nD[0]
                # print('v{}={}'.format(j, ratings_j))
                # print("-----")
                self.sim_arr[i, j] = self.dist_func(
                    ratings_i, ratings_j)
            self.sim_arr[i]
        self.S = self.sim_arr

    def loop_userzzzzzz(self):
        users = self.Y_data[:, 0]
        self.sim_arr = np.zeros((self.n_users, self.n_users))
        for i in range(self.n_users):
            ids = np.where(users == i)[0].astype(np.int32)
            # item_ids = self.Y_data[ids, 1].astype(np.int32)
            # item_ids = np.array([])
            item_ids = self.Y_data[ids, 1].astype(np.int32)
            # print("item_ids", item_ids)
            ratings_i = self.Ybar[i, :]
            # print("ratings_i", ratings_i)
            print('u{}'.format(i))
            for j in range(self.n_users):
                jds = np.where(users == j)[0].astype(np.int32)
                item_jds = self.Y_data[jds, 1].astype(np.int32)
                common = np.intersect1d(item_ids, item_jds)
                # bắt thưa
                ratings_j = self.Ybar[j, :]
                self.sim_arr[i, j] = self.dist_func(
                    common, ratings_i, ratings_j)
        self.S = self.sim_arr

    def loop_usernnn(self):
        users = self.Y_data[:, 0]
        self.sim_arr = np.zeros((self.n_users, self.n_users))
        for i in range(self.n_users):
            ids = np.where(users == i)[0].astype(np.int32)
            # item_ids = self.Y_data[ids, 1].astype(np.int32)
            # item_ids = np.array([])
            item_ids = self.Y_data[ids, 1].astype(np.int32)
            # print("item_ids", item_ids)
            ratings_i = self.Ybar[i, :]
            # print("ratings_i", ratings_i)
            print('u{}'.format(i))
            for j in range(self.n_users):
                jds = np.where(users == j)[0].astype(np.int32)
                item_jds = self.Y_data[jds, 1].astype(np.int32)
                # bắt thưa
                ratings_j = self.Ybar[j, :]
                self.sim_arr[i, j] = self.dist_func(ratings_i, ratings_j)
        self.S = self.sim_arr

    # TÍNH NGƯỠNG
    '''
    def calc_personal_thes(u):
        rbar_u = self.mu[u]
        rbar_global = np.mean(self.Y_data[:, 2])
        rbar_scale = 3
        return (rbar_u + rbar_global + rbar_scale) / 3'''

    # tính độ tương đồng

    def similarity(self, method=1):
        if method == 1:  # cosine
            # T là ma trận nghịch đảo, ở đây nghịch đảo vị trí user-item thành vị trí item-user
            self.S = self.dist_func(self.Ybar, self.Ybar)  # dùng thư viện
            # self.loop_user()
            # jaccard np.where(a > 0.5, 1, 0) ý tưởng là dùng hàm trên id của item non zero
        if method == 2:
            self.sim_jaccard = np.zeros((self.n_users, self.n_users))
            users = self.Y_data[:, 0]
            # print(self.sim_jaccard[0, 1])
            for i in range(self.n_users):
                ids = np.where(users == i)[0].astype(np.int32)
                item_ids = self.Y_data[ids, 1]
                # print('u{}={}'.format(i, item_ids))
                print('u{}'.format(i))
                # self.sim_jaccard[ids, ids] = 1
                for j in range(self.n_users):
                    jds = np.where(users == j)[0].astype(np.int32)
                    item_jds = self.Y_data[jds, 1]
                    # print('v{}={}'.format(j, item_jds))
                    # print("-----")
                    self.sim_jaccard[i, j] = self.dist_func(
                        item_ids, item_jds)
            self.S = self.sim_jaccard
        if method == 3:  # urp
            users = self.Y_data[:, 0]
            self.sim_arr = np.zeros((self.n_users, self.n_users))
            for i in range(self.n_users):
                ids = np.where(users == i)[0].astype(np.int32)
            # item_ids = self.Y_data[ids, 1].astype(np.int32)
            # item_ids = np.array([])
                item_ids = self.Y_data[ids, 1].astype(np.int32)
            # print("item_ids", item_ids)
                ratings_i = self.Ybar[i, :]
            # print("ratings_i", ratings_i)
                print('u{}'.format(i))
                for j in range(self.n_users):
                    jds = np.where(users == j)[0].astype(np.int32)
                    item_jds = self.Y_data[jds, 1].astype(np.int32)
                # bắt thưa
                    ratings_j = self.Ybar[j, :]
                    self.sim_arr[i, j] = self.dist_func(ratings_i, ratings_j)
            self.S = self.sim_arr
        if method == 4:  # COR
            self.loop_userzzzzzz()
        if method == 5:  # CPC
            users = self.Y_data[:, 0]
            self.sim_arr = np.zeros((self.n_users, self.n_users))
            for i in range(self.n_users):
                ids = np.where(users == i)[0].astype(np.int32)
            # item_ids = self.Y_data[ids, 1].astype(np.int32)
            # item_ids = np.array([])
                item_ids = self.Y_data[ids, 1].astype(np.int32)
            # print("item_ids", item_ids)
                ratings_i = self.Ybar[i, :]
            # print("ratings_i", ratings_i)
                print('u{}'.format(i))
                for j in range(self.n_users):
                    jds = np.where(users == j)[0].astype(np.int32)
                    item_jds = self.Y_data[jds, 1].astype(np.int32)
                    common = np.intersect1d(item_ids, item_jds)
                # bắt thưa
                    ratings_j = self.Ybar[j, :]
                    self.sim_arr[i, j] = self.dist_func(
                        common, item_ids, item_jds, ratings_i, ratings_j)
            self.S = self.sim_arr

        if method == 6:  # WPC
            users = self.Y_data[:, 0]
            self.sim_arr = np.zeros((self.n_users, self.n_users))
            mean_global = np.mean(self.Y_data[:, 2])
            for i in range(self.n_users):
                ids = np.where(users == i)[0].astype(np.int32)
            # item_ids = self.Y_data[ids, 1].astype(np.int32)
            # item_ids = np.array([])
                item_ids = self.Y_data[ids, 1].astype(np.int32)
            # print("item_ids", item_ids)
                ratings_i = self.Ybar[i, :]
                thres_i = (mean_global + 3 + self.mu[i])/3
            # print("ratings_i", ratings_i)
                print('u{}'.format(i))
                for j in range(self.n_users):
                    thres_j = (mean_global + 3 + self.mu[j])/3
                    jds = np.where(users == j)[0].astype(np.int32)
                    item_jds = self.Y_data[jds, 1].astype(np.int32)
                    common = np.intersect1d(item_ids, item_jds)
                # bắt thưa
                    ratings_j = self.Ybar[j, :]
                    self.sim_arr[i, j] = self.dist_func(
                        common, item_ids, item_jds, ratings_i, ratings_j, thres_i, thres_j)
            self.S = self.sim_arr
        if method == 7:
            users = self.Y_data[:, 0]
            self.sim_arr = np.zeros((self.n_users, self.n_users))
            for i in range(self.n_users):
                ids = np.where(users == i)[0].astype(np.int32)
                # item_ids = self.Y_data[ids, 1].astype(np.int32)
                # item_ids = np.array([])
                item_ids = self.Y_data[ids, 1].astype(np.int32)
                # print("item_ids", item_ids)
                ratings_i = self.Ybar[i, :]
                # print("ratings_i", ratings_i)
                print('u{}'.format(i))
                for j in range(self.n_users):
                    jds = np.where(users == j)[0].astype(np.int32)
                    item_jds = self.Y_data[jds, 1].astype(np.int32)
                    common = np.intersect1d(item_ids, item_jds)
                    mean_common = self.mean_rating_item_i(common)
                    # bắt thưa
                    ratings_j = self.Ybar[j, :]
                    self.sim_arr[i, j] = self.dist_func(common, item_ids, item_jds,
                                                        ratings_i, ratings_j, mean_common)
            self.S = self.sim_arr

    def mean_rating_item_i1(self, i_arr):
        mean_ri_arr = np.array([])
        for id_i in range(i_arr.size):
            mean_ri = np.mean(self.Ybar[:, i_arr[id_i]].data)
            mean_ri_arr = np.append(mean_ri_arr, [mean_ri], axis=0)
        return mean_ri_arr

    def mean_rating_item_i(self, i_arr):
        mean_ri_arr = np.array([])
        for id_i in i_arr:
            mean_ri = np.mean(self.Ybar[:, id_i].data)
            mean_ri_arr = np.append(mean_ri_arr, [mean_ri], axis=0)
        return mean_ri_arr
    # Thực hiện lại 2 hàm phía trên khi có thêm dữ liệu:

    def refresh(self, method=1):
        """
        Normalize data and calculate similarity matrix again (after
        some few ratings added)
        """
        self.normalize_Yzzzzzzz()
        self.similarity(method)

    def fit(self, method=1):
        self.refresh(method)

    # Predict và recommend

    def __pred(self, u, i, normalized=1):
        sim = self.S[u, :]
        a = np.argsort(sim)[-self.k:]
        nearest_s = sim[a]
        ids = np.where(self.Y_data[:, 1] == i)[0].astype(np.int32)
        users_rated_i = (self.Y_data[ids, 0]).astype(np.int32)
        # rating_users_i = (self.Y_data[ids, 2]).astype(np.int32)
        # print("a", a)
        # print("users_rated_i", users_rated_i)

        # tìm các user trong có đánh giá item i
        ari = np.array([])
        for j in range(a.size):
            if a[j] in users_rated_i:
                ari = np.append(ari, [a[j]], axis=0)
        ari = ari.astype(np.int32)
        nearest_s_ari = sim[ari]
        r = self.Ybar[ari, i].data - self.mu[ari]
        if (ari.size == 0):
            return 0, self.mu[u]
        else:
            p = (r*nearest_s_ari).sum() / \
                (np.abs(nearest_s_ari).sum() + 1e-8) + self.mu[u]
            return p, self.mu[u]

    def __pred1(self, u, i, normalized=1):
        """
        predict the rating of user u for item i (normalized)
        if you need the un
        """
        # Step 1: find all users who rated i
        # ids: vị trí của item i trong mảng Y_data
        ids = np.where(self.Y_data[:, 1] == i)[0].astype(np.int32)
        # Step 2:
        users_rated_i = (self.Y_data[ids, 0]).astype(np.int32)
        rating_users_i = (self.Y_data[ids, 2]).astype(np.int32)
        # Step 3: find similarity btw the current user and others
        # who already rated i
        sim = self.S[u, users_rated_i]
        # Step 4: find the k most similarity users
        a = np.argsort(sim)[-self.k:]
        # and the corresponding similarity levels
        nearest_s = sim[a]
        # How did each of 'near' users rated item i
        # r = self.Y[i, users_rated_i[a]].toarray()
        r = rating_users_i[a]-self.mu[users_rated_i[a]]
        '''print("tuong dong", self.S[0, 108])
        print("users_rated_i", users_rated_i)
        print("mang y", self.Y[16, 108])
        print("mu[4] nè: ", self.mu[108])
        print("mang ybar", self.Ybar[16, 108])
        print(" ng a nè", users_rated_i[a])
        print("sản phẩm nè", i)
        print("u nè", u)
        print("a nè:", a)
        print("rating của ng a nè", rating_users_i[a])
        print("mu[a] nè: ", self.mu[users_rated_i[a]])
        print("r nè:", r)
        print("nearest_s nè:", nearest_s) '''
        # print("r", r)
        # print("(nearest_s)", nearest_s)
        '''print("pred:", (r*nearest_s).sum() /
              (np.abs(nearest_s).sum() + 1e-8) + self.mu[u])'''
        # add a small number, for instance, 1e-8, to avoid dividing by 0
        return (r*nearest_s).sum()/(np.abs(nearest_s).sum() + 1e-8) + self.mu[u]

    def pred(self, u, i, normalized=1):
        """
        predict the rating of user u for item i (normalized)
        if you need the un
        """
        if self.uuCF:
            return self.__pred(u, i, normalized)
        return self.__pred(i, u, normalized)

    def recommend(self, u):
        """
        Determine all items should be recommended for user u.
        The decision is made based on all i such that:
        self.pred(u, i) > 0. Suppose we are considering items which
        have not been rated by u yet.
        """
        ids = np.where(self.Y_data[:, 0] == u)[0]
        items_rated_by_u = self.Y_data[ids, 1].tolist()
        recommended_items = []
        sim = self.S[u, :]
        a = np.argsort(sim)[-self.k:]
        nearest_s = sim[a]
        for i in range(self.n_items):
            if i not in items_rated_by_u and nearest_s.size > 0:
                rating = self.__pred(u, i)
                if rating > 0:
                    recommended_items.append(i)
            else:
                return 0

        return recommended_items

    def recommend2(self, u):
        """
        Determine all items should be recommended for user u.
        The decision is made based on all i such that:
        self.pred(u, i) > 0. Suppose we are considering items which
        have not been rated by u yet.
        """
        ids = np.where(self.Y_data[:, 0] == u)[0]
        items_rated_by_u = self.Y_data[ids, 1].tolist()
        recommended_items = []

        for i in range(self.n_items):
            if i not in items_rated_by_u:
                rating = self.__pred(u, i)
                if rating > 0:
                    recommended_items.append(i)

        return recommended_items

    def print_recommendation(self):
        """
        print all items which should be recommended for each user
        """
        print('Recommendation: ')
        for u in range(self.n_users):
            recommended_items = self.recommend(u)
            if self.uuCF:
                print('    Recommend item(s):',
                      recommended_items, 'for user', u)
            else:
                print('    Recommend item', u,
                      'for user(s) : ', recommended_items)
