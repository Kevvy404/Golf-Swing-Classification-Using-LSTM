import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load your data
file_path = 'Data/extracted_features.csv'
data = pd.read_csv(file_path)

# print("Data shape:", data.shape)

# Proceed with PCA only if there are enough samples and features
if data.shape[0] > 1 and data.shape[1] > 1:
    # Choose the number of components to be the minimum of number of samples or features minus one
    n_components = min(data.shape[0], data.shape[1]) - 1
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data)
    # print("PCA completed with components:", n_components)
    # print("Variance Ratio:", pca.explained_variance_ratio_)
else:
    print("Not enough data to perform PCA. Need more than one sample and one feature.")


# Assuming `pca` is your PCA object and `data` is your DataFrame used for PCA
loadings = pd.Series(pca.components_[0], index=data.columns)
# print(loadings.sort_values(ascending=False))

plt.figure(figsize=(10, 7))
plt.scatter(principal_components[:, 0], principal_components[:, 1], alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Projection')
plt.show()

