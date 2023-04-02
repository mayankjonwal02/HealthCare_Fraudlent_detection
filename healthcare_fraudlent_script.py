# %% [markdown]
# ## Importing data
# 

# %%
import pandas as pd
import numpy as np

data = pd.read_csv("Healthcare Providers.csv")
data.head()
data.shape


# %%
data.columns

# %% [markdown]
# ## checking value counts
# 

# %%
data.describe()

# %%
for i in data.columns:
    print(i)
    print(data[i].value_counts())

# %% [markdown]
# ## Pre-processing

# %%
dropcols =['index', 'National Provider Identifier',
       'Last Name/Organization Name of the Provider',
       'First Name of the Provider', 'Middle Initial of the Provider','Credentials of the Provider',
       'Street Address 1 of the Provider',
       'Street Address 2 of the Provider','City of the Provider','Zip Code of the Provider', 'State Code of the Provider',
       'Country Code of the Provider','HCPCS Description','Entity Type of the Provider']

# %%
data = data.drop(dropcols, axis=1)
data.describe()


# %%
print(data.isnull().sum())
data = data.dropna()
print("after dropping null values")
print(data.isnull().sum())

# %%
data.describe()

# %%
data.dtypes

# %% [markdown]
# ## conversion of datatypes

# %%
def convert_series_to_float(series):
    # Apply a function to each element of the Series
    return series.apply(lambda x: float(str(x).replace(',', '')))

# %%
floatcols = ['Number of Services',
       'Number of Medicare Beneficiaries',
       'Number of Distinct Medicare Beneficiary/Per Day Services',
       'Average Medicare Allowed Amount', 'Average Submitted Charge Amount',
       'Average Medicare Payment Amount',
       'Average Medicare Standardized Amount']

for i in floatcols:
    data[i] = convert_series_to_float(data[i])   

# %%
for i in floatcols:
    for j in data[i]:
        print(type(j))

# %%
data.dtypes

# %% [markdown]
# ## Categorical Encoding

# %%

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()


# %%
for i in data.columns:
    if data[i].dtype == 'object':
        data[i] = le.fit_transform(data[i])

# %%
data

# %%
data.dtypes

# %%
data.describe()

# %%
# # Compute the z-scores for each feature
# z_scores = np.abs((data - data.mean()) / data.std())

# # Define a threshold z-score for outlier detection
# threshold = 3

# # Create a boolean mask of the outlier rows
# outliers = (z_scores > threshold).any(axis=1)

# # Remove the outlier rows from the data
# data = data[~outliers]

# %%
data.describe()

# %% [markdown]
# ## standardization

# %%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Extract the columns to be scaled
cols_to_scale = data.columns[6:]


# Standardize the selected columns
data.loc[:, cols_to_scale] = scaler.fit_transform(data.loc[:, cols_to_scale])

data.head()

# %%
data.columns[6:]

# %% [markdown]
# ## Feature selection

# %%
len(data.columns)

# %%
from sklearn.decomposition import PCA

pca = PCA(0.95)  # retaining 95% of useful features

data_pca = pca.fit_transform(data)

data_pca.shape


# %% [markdown]
# not worthy as got only one use feature

# %%
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt




# Train KMeans clustering
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(data)




# # # Compute silhouette scores
kmeans_score = silhouette_score(data, kmeans_labels)
# # agg_score = silhouette_score(data, agg_labels)

# # # Print the silhouette scores
print(f'KMeans silhouette score: {kmeans_score}')
# # print(f'AgglomerativeClustering silhouette score: {agg_score}')

# # # Visualize the clusters
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].scatter(data.iloc[:, 0], data.iloc[:, 1], c=kmeans_labels)
axs[0].set_title(f'KMeans (score: {kmeans_score:.2f})')
# # axs[1].scatter(data.iloc[:, 0], data.iloc[:, 1], c=agg_labels)
# # axs[1].set_title(f'AgglomerativeClustering (score: {agg_score:.2f})')


# %%
# Train DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(data)
dbscan_score = silhouette_score(data, dbscan_labels)
print(f'DBSCAN silhouette score: {dbscan_score}')
axs[2].scatter(data.iloc[:, 0], data.iloc[:, 1], c=dbscan_labels)
axs[2].set_title(f'DBSCAN (score: {dbscan_score:.2f})')
plt.show()


# %%
# Choose the number of clusters
num_clusters = 3

# Train the K-means clustering algorithm
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(data)

# Identify outliers
cluster_counts = pd.Series(kmeans.labels_).value_counts()
outlier_clusters = cluster_counts[cluster_counts < cluster_counts.quantile(0.05)].index.tolist()

outliers = []
for i in outlier_clusters:
    outliers += list(data[kmeans.labels_ == i].index)

print(len(outliers))

# Evaluate the results
for outlier in outliers:
    # Investigate the outlier to determine if it is truly fraudulent activity or false positive
    print(f"Potential outlier detected at index {outlier}")
    print(data.loc[outlier])

# %%
# kmeans_labels = kmeans.fit_predict(data-data.loc[outlier])




# # # # Compute silhouette scores
# kmeans_score = silhouette_score(data, kmeans_labels)

outliers

# %%
data = data.drop(index=outliers)

# %%
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(data)




# # # Compute silhouette scores
kmeans_score = silhouette_score(data, kmeans_labels)
# # agg_score = silhouette_score(data, agg_labels)

# # # Print the silhouette scores
print(f'KMeans silhouette score: {kmeans_score}')
# # print(f'AgglomerativeClustering silhouette score: {agg_score}')

# # # Visualize the clusters
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].scatter(data.iloc[:, 0], data.iloc[:, 1], c=kmeans_labels)
axs[0].set_title(f'KMeans (score: {kmeans_score:.2f})')

# %%


