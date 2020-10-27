from classCF import *
from sklearn.metrics import f1_score
import plotly.graph_objects as go
# data file user-user CF:
r_cols = ['user_id', 'item_id', 'rating']
ratings_base = pd.read_csv('vi_du.dat', sep='\t',
                           names=r_cols, encoding='latin-1')
rate_train = ratings_base.to_numpy()  # as_matrix() không chạy
print('rate_train', rate_train)

COS = cos_similarity  # method = 4
COS_lib = cosine_similarity  # method = 1
JAC = jaccard_similarity  # method = 2
MSD = msd_similarity  # method = 4
MSDPP = msd_similarity_pp  # method = 4
COR = cor_similarity  # method = 4
CPC = cpc_similarity  # method = 4
RJ = rjaccard_similarity  # method = 2
RJMSD = rjmsd_similarity  # method = 5
JMSD = jmsd_similarity  # method = 5
URP = urp_similarity  # method = 3
PIP = pip_similarity  # method = 7
RPB = rpb_similarity  # method = 4
RATEJ = rating_jaccard_similarity  # method = 4
RATEJRPB = rating_jaccard_rpb_similarity  # method = 4
JACLMH = jaclmh_similarity  # method = 4
PSS = pss_similarity  # method = 7
JPSS = jpss_similarity  # method = 7
NHSM = nhsm_similarity  # method = 7
TMJ = tmj_similarity  # method = 5
CTJ = ctj_similarity  # method = 5
JUOD = jacuod_similarity  # method = 5
JLMHUOD = jlmhuod_similarity  # method = 4
JSMCC = jsmcc_similarity  # method = 5
rs = CF(rate_train, k=2, dist_func=RATEJRPB, uuCF=1)
rs.fit(4)
print(rs.Ybar.todense())
print(rs.S)
Sim = rs.S.round(3)
fig = go.Figure(data=[go.Table(
    header=dict(values=['', 'u0', 'u1', 'u2', 'u3', 'u4'],
                line_color='darkslategray',
                fill_color='lightgrey',
                align='center'),
    cells=dict(values=[['u0', 'u1', 'u2', 'u3', 'u4'], Sim[:, 0], Sim[:, 1], Sim[:, 2], Sim[:, 3], Sim[:, 4]],
               line_color='darkslategray',
               fill_color=['lightgrey', 'white'],
               align='center'))
])
fig.write_html('first_figure.html', auto_open=True)
fig.show()
