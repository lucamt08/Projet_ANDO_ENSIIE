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

plt.figure(figsize=(6,4))
sns.countplot(data=alz_df, x='FamilyHistoryAlzheimers', hue='Diagnosis', palette='coolwarm')
plt.title('Family History vs Alzheimer’s Diagnosis')
plt.xlabel('Family History of Alzheimer’s (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.legend(title='Diagnosis', labels=['Non-Diagnosed', 'Diagnosed'])
plt.show()


# Calculate correlations with Diagnosis
def analyze_correlations(df, target_col='Diagnosis'):
    # Create a copy to avoid modifying original data
    temp_df = df.copy()
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_cols = temp_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != target_col:
            temp_df[col] = le.fit_transform(temp_df[col].astype(str))
    
    # Calculate correlations with target
    correlations = temp_df.corr()[target_col].abs().sort_values(ascending=False)
    
    # Plot correlations
    plt.figure(figsize=(10, 8))
    correlations[1:].plot(kind='barh')  # Exclude Diagnosis itself
    plt.title('Feature Correlations with Alzheimer\'s Diagnosis')
    plt.xlabel('Absolute Correlation')
    plt.tight_layout()
    plt.show()
    
    return correlations

correlations = analyze_correlations(alz_df)
print("Top 10 correlated features:")
print(correlations.head(11))  # Includes Diagnosis itself





# Define variable categories
# demographic_vars = ['Age', 'Gender', 'Ethnicity', 'EducationLevel']
# lifestyle_vars = ['BMI', 'Smoking', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality']
# medical_history_vars = ['FamilyHistoryAlzheimers', 'CardiovascularDisease', 'Diabetes', 'Depression', 
#                        'HeadInjury', 'Hypertension']
# vital_signs_vars = ['SystolicBP', 'DiastolicBP', 'CholesterolTotal', 'CholesterolLDL', 
#                    'CholesterolHDL', 'CholesterolTriglycerides']
# cognitive_vars = ['MMSE', 'FunctionalAssessment', 'MemoryComplaints', 'BehavioralProblems', 
#                  'ADL', 'Confusion', 'Disorientation', 'PersonalityChanges', 
#                  'DifficultyCompletingTasks', 'Forgetfulness']
# target_var = ['Diagnosis']

# # Create correlation matrices for each category
# categories = {
#     'Demographic Variables': demographic_vars + target_var,
#     'Lifestyle Variables': lifestyle_vars + target_var,
#     'Medical History': medical_history_vars + target_var,
#     'Vital Signs & Cholesterol': vital_signs_vars + target_var,
#     'Cognitive & Functional Assessment': cognitive_vars
# }

# for category_name, variables in categories.items():
#     # Filter variables that exist in the dataframe
#     existing_vars = [var for var in variables if var in alz_df.columns]
    
#     if len(existing_vars) > 1:  # Need at least 2 variables to correlate
#         plt.figure(figsize=(12, 10))
#         corr_matrix = alz_df[existing_vars].corr()
        
#         sns.heatmap(corr_matrix, 
#                    annot=True, 
#                    fmt='.2f', 
#                    cmap='coolwarm', 
#                    center=0,
#                    square=True,
#                    cbar_kws={'shrink': 0.8})
#         plt.title(f'Correlation Matrix: {category_name}', fontsize=14, fontweight='bold')
#         plt.tight_layout()
#         plt.show()