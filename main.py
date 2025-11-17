import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import seaborn as sns

alz_df = pd.read_csv("alzheimer_data.csv")
alz_df.drop('DoctorInCharge', axis=1, inplace=True)
print(alz_df.head())

print(alz_df.info())

non_diagnosed_count = len(alz_df.loc[alz_df['Diagnosis'] == 0].index)
diagnosed_count = len(alz_df.loc[alz_df['Diagnosis'] == 1].index)

print(f"Number of observations : {len(alz_df.index)}")
print(f"Non Diagnosed cases : {non_diagnosed_count}\nDiagnosed cases : {diagnosed_count}")


X = alz_df.drop(columns=['Diagnosis']).values
y = alz_df['Diagnosis'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Perform feature scaling
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


pca = PCA(n_components=2)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

explained_variance = pca.explained_variance_ratio_

print(f"Explained variance by each principal component: {explained_variance}")


plt.figure(figsize=(8,6))
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], alpha=0.7, label='Non-Diagnosed', edgecolors='k')
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], alpha=0.7, label='Diagnosed', edgecolors='k')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA - Alzheimer’s Data')
plt.legend()
plt.show()

# Diagnosis Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Diagnosis', hue='Diagnosis', data=alz_df, palette='viridis', legend=False)
plt.title('Diagnosis Distribution (0: No Alzheimer\'s, 1: Alzheimer\'s)', fontsize=16)
plt.xlabel('Diagnosis', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks([0, 1], ['No Alzheimer\'s', 'Alzheimer\'s'])
plt.show()


# Correlation Heatmap
corr_matrix=alz_df.corr()
plt.figure(figsize=(30,15))
sns.heatmap(corr_matrix, annot=True, annot_kws={"size": 5})
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
sns.boxplot(x=alz_df['Diagnosis'], y=alz_df['ADL'], data=alz_df)
plt.show()



