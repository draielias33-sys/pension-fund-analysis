import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import numpy as np
from sklearn.mixture import BayesianGaussianMixture

df = pd.read_csv('pensia-net1999-2022.csv', encoding='utf-8')
print(len(df))
print(df.columns)
# understand how many pension fund we have
print(df['FUND_ID'].nunique())
# understand how many Fund have only NaN value at the columns 'AVG_ANNUAL_YIELD_TRAILING_3YRS'
print(df.groupby('FUND_ID')['AVG_ANNUAL_YIELD_TRAILING_3YRS'].apply(lambda x: x.isna().all()).sum())
# delete the NaN raws
df = df.dropna(subset=['AVG_ANNUAL_YIELD_TRAILING_3YRS', 'AVG_DEPOSIT_FEE', 'SHARPE_RATIO', 'STOCK_MARKET_EXPOSURE'])
print(len(df))
# checking how many funds left
print(df['FUND_ID'].nunique())
# choose the variables for our algorithm, these variables describes the objective performance of the fund
features = [
    'AVG_ANNUAL_YIELD_TRAILING_3YRS', 'AVG_DEPOSIT_FEE',
    'SHARPE_RATIO',
    'STOCK_MARKET_EXPOSURE'
]
# aggregate by FUND_ID
df_fund = (
    df
    .groupby(['FUND_ID', 'FUND_NAME'])[features]
    .mean()
    .reset_index()
)
# checking the aggregation
print(df_fund.shape)
print(df_fund['FUND_ID'].nunique())
# BEGIN THE GMM ALGORITHM

# build the X matrix
X = df_fund[features]

# normalise

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# choose the optimal amount of cluster
bic = []
K = range(1, 8)

for k in K:
    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(X_scaled)
    bic.append(gmm.bic(X_scaled))

best_k = K[np.argmin(bic)]
print("Best amount of cluster :", best_k)
# apply GMM
gmm = GaussianMixture(n_components=best_k, random_state=42)
df_fund['cluster'] = gmm.fit_predict(X_scaled)

# see the result
print(df_fund.groupby('cluster')[features].mean().to_string())

# checking the len of each cluster
print(df_fund['cluster'].value_counts().to_string())

# we can see an excellent sharpe ratio for the cluster 2. (2.134)
df_cluster2 = df_fund[df_fund['cluster'] == 2]
# we choose, from cluster 2, the 5 best sharpe ratio :
top_pension_funds = df_cluster2[['FUND_NAME', 'SHARPE_RATIO']].sort_values('SHARPE_RATIO', ascending=False)
print(top_pension_funds)
# we can decide which pension funds are the best based on sharpe ratio
print(f' The 5 best pension fund with GMM algo are : \n {(top_pension_funds['FUND_NAME'].head(5))}')

# BEGIN THE BGMM ALGORITHM

bgmm = BayesianGaussianMixture(
    n_components=8,  # K MAXIMAL
    covariance_type='full',
    weight_concentration_prior_type='dirichlet_process',
    weight_concentration_prior=0.5,  # encourage peu de clusters
    n_init=5,
    random_state=42
)

df_fund['cluster_bgmm'] = bgmm.fit_predict(X_scaled)

weights = bgmm.weights_
print("Weights of clusters :")
for i, w in enumerate(weights):
    print(f"Cluster {i}: {w:.4f}")

print(
    df_fund
    .groupby('cluster_bgmm')[features]
    .mean()
    .to_string()
)

proba_bgmm = bgmm.predict_proba(X_scaled)

df_fund['bgmm_confidence'] = proba_bgmm.max(axis=1)

# the best cluster seems to be the n2
print(
    df_fund['cluster_bgmm']
    .value_counts()
    .sort_index()
    .to_string()
)

cluster_id = 2


best_pension_fund_BGMM =(df_fund[df_fund['cluster_bgmm'] == cluster_id]
    [['FUND_NAME', 'SHARPE_RATIO']]
    .sort_values('SHARPE_RATIO', ascending=False))


print(f'the best 5 pension fund with BGMM algo are : {best_pension_fund_BGMM.head(5)}')

#verify the relevant of the algos

from sklearn.metrics import silhouette_score

sil_gmm = silhouette_score(X_scaled, df_fund['cluster'])
sil_bgmm = silhouette_score(X_scaled, df_fund['cluster_bgmm'])

print("Silhouette GMM:", sil_gmm)
print("Silhouette BGMM:", sil_bgmm)

#let's see if the best funds are the most popular ones


df_popularity = (df.groupby(['FUND_ID', 'FUND_NAME'])['DEPOSITS'].mean().reset_index().
                 sort_values(by='DEPOSITS', ascending=False)) # Bonus : trier par le plus populaire
print(df_popularity)




