#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[ ]:


import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, precision_score, recall_score, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')


# In[ ]:


def load_data(file_path):
    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape}")
    return df


# In[ ]:


# Exploratory Data Analysis
def perform_eda(df):
    print("\nBasic Information:")
    print(df.info())
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    

    print("\nTarget Variable Distribution:")
    print(df['FraudFound'].value_counts(normalize=True) * 100)
    
    plt.figure(figsize=(8, 6))
    sns.countplot(x='FraudFound', data=df)
    plt.title('Distribution of Fraud')
    plt.savefig('fraud_distribution.png')
    
    print("\nFeature Correlations with Target:")
    for col in df.select_dtypes(include=['number']).columns:
        if col != 'FraudFound':
            print(f"{col}: {df[col].corr(df['FraudFound'])}")
    
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'FraudFound':
            plt.figure(figsize=(12, 6))
            fraud_rate = df.groupby(col)['FraudFound'].mean()
            fraud_rate.sort_values().plot(kind='bar')
            plt.title(f'Fraud Rate by {col}')
            plt.ylabel('Fraud Rate')
            plt.tight_layout()
            plt.savefig(f'fraud_rate_by_{col}.png')
    
    return


# In[ ]:


def preprocess_data(df):
    if df['FraudFound'].dtype == 'object':
        df['FraudFound'] = df['FraudFound'].map({'Yes': 1, 'No': 0})
    
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    if 'FraudFound' in categorical_features:
        categorical_features.remove('FraudFound')
    
    numerical_features = df.select_dtypes(include=['number']).columns.tolist()
    if 'FraudFound' in numerical_features:
        numerical_features.remove('FraudFound')
    
    for col in ['PastNumberOfClaims', 'NumberOfSuppliments']:
        if col in df.columns and df[col].dtype == 'object' and col in numerical_features:
            numerical_features.remove(col)
            categorical_features.append(col)
    
    special_columns = []
    for col in df.columns:
        if col in ['Days:Policy-Accident', 'Days:Policy-Claim'] and df[col].dtype == 'object':
            special_columns.append(col)
            categorical_features.append(col)
    for col in special_columns:
        if "more than" in df[col].iloc[0].lower():
            df = df.drop(col, axis=1)
            if col in numerical_features:
                numerical_features.remove(col)
            if col in categorical_features:
                categorical_features.remove(col)
    
    for col in ['Age']:
        if col in df.columns and col in categorical_features:
            if 'to' in df[col].iloc[0]:
                df = df.drop(col, axis=1)
                categorical_features.remove(col)
            elif 'more than' in str(df[col].iloc[0]).lower():
                df[col] = df[col].apply(lambda x: float(str(x).lower().replace('more than ', '')) if isinstance(x, str) and 'more than' in str(x).lower() else float(x))
                categorical_features.remove(col)
                numerical_features.append(col)
    
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return df, preprocessor


# In[ ]:


# Split the data
def split_data(df, preprocessor):
    X = df.drop('FraudFound', axis=1)
    y = df['FraudFound']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    print(f"Training set shape: {X_train_processed.shape}")
    print(f"Testing set shape: {X_test_processed.shape}")
    
    return X_train, X_test, y_train, y_test


# In[ ]:


# Evaluate model performance
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    
    if isinstance(model, keras.Model):
        y_pred = (y_pred > 0.5).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n----- {model_name} Performance -----")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'confusion_matrix_{model_name}.png')
    
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif isinstance(model, keras.Model):
        y_proba = model.predict(X_test).ravel()
    else:
        y_proba = y_pred  #
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    plt.savefig(f'roc_curve_{model_name}.png')
    
    return {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': roc_auc if 'roc_auc' in locals() else None
    }


# In[ ]:


# Train classical ML models
def train_classical_models(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'Decision Tree': DecisionTreeClassifier(class_weight='balanced'),
        'Random Forest': RandomForestClassifier(class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(),
        'SVM': SVC(probability=True, class_weight='balanced')
    }
    
    results = []
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Evaluate
        metrics = evaluate_model(model, X_test, y_test, name)
        results.append(metrics)
        
        if name in ['Decision Tree', 'Random Forest', 'Gradient Boosting']:
            feature_importances = None
            if hasattr(model, 'feature_importances_'):
                feature_importances = model.feature_importances_
            
            if feature_importances is not None:
                try:
                    feature_names = preprocessor.get_feature_names_out()
                except:
                    feature_names = [f"feature_{i}" for i in range(len(feature_importances))]
                
                top_n = 15
                indices = np.argsort(feature_importances)[-top_n:]
                plt.figure(figsize=(10, 8))
                plt.barh(range(top_n), feature_importances[indices])
                plt.yticks(range(top_n), [feature_names[i] for i in indices])
                plt.xlabel('Feature Importance')
                plt.title(f'Top {top_n} Feature Importance - {name}')
                plt.tight_layout()
                plt.savefig(f'feature_importance_{name}.png')
    
    return results, models


# In[ ]:


# Build and train neural network
def train_neural_network(X_train, X_test, y_train, y_test):
    input_dim = X_train.shape[1]
    
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_dim=input_dim),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    neg, pos = np.bincount(y_train.astype(int))
    weight_for_0 = (1 / neg) * (len(y_train) / 2.0)
    weight_for_1 = (1 / pos) * (len(y_train) / 2.0)
    class_weight = {0: weight_for_0, 1: weight_for_1}
    
    print("\nTraining Neural Network...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        class_weight=class_weight
    )
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.tight_layout()
    plt.savefig('nn_training_history.png')
    
    metrics = evaluate_model(model, X_test, y_test, 'Neural Network')
    
    return metrics, model


# In[ ]:


def compare_models(results):
    comparison_df = pd.DataFrame(results)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    plt.figure(figsize=(15, 10))
    
    for i, metric in enumerate(metrics):
        if metric in comparison_df.columns:
            plt.subplot(2, 3, i+1)
            sns.barplot(x='model', y=metric, data=comparison_df)
            plt.title(f'Model Comparison - {metric.capitalize()}')
            plt.xticks(rotation=45)
            plt.tight_layout()
    
    plt.savefig('model_comparison.png')
    
    print("\nBest Models by Metric:")
    for metric in metrics:
        if metric in comparison_df.columns:
            best_model = comparison_df.loc[comparison_df[metric].idxmax()]
            print(f"Best model for {metric}: {best_model['model']} ({best_model[metric]:.4f})")
    
    return comparison_df


# In[ ]:


# Feature Engineering
def feature_engineering(df):
    
    if all(x in df.columns for x in ['Month', 'MonthClaimed']):
        month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 
                     'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
        
        df['Month_Num'] = df['Month'].map(month_map)
        df['MonthClaimed_Num'] = df['MonthClaimed'].map(month_map)
        
        df['MonthDiff'] = ((df['MonthClaimed_Num'] - df['Month_Num']) % 12).abs()
    
    if 'DayOfWeek' in df.columns:
        df['Weekend_Accident'] = df['DayOfWeek'].isin(['Saturday', 'Sunday']).astype(int)
    
    if 'DayOfWeekClaimed' in df.columns:
        df['Weekend_Claim'] = df['DayOfWeekClaimed'].isin(['Saturday', 'Sunday']).astype(int)
    
    if all(x in df.columns for x in ['Age',]):
        if df['Age'].dtype != 'float64' and df['Age'].dtype != 'int64':
            if 'to' in str(df['Age'].iloc[0]):
                df['Age'] = df['Age'].apply(lambda x: np.mean([float(i) for i in str(x).replace('to', '-').replace(' ', '').split('-')]) if isinstance(x, str) else x)
        
        
    
    if all(x in df.columns for x in ['Days:Policy-Accident', 'Days:Policy-Claim']):
        for col in ['Days:Policy-Accident', 'Days:Policy-Claim']:
            if df[col].dtype == 'object':
                if 'more than' in str(df[col].iloc[0]):
                    df[col] = df[col].apply(lambda x: float(str(x).replace('more than ', '')) if 'more than' in str(x) else 
                                          (float(str(x).replace('less than ', '')) if 'less than' in str(x) else float(x)))
        
        df['Days_Accident_To_Claim'] = abs(df['Days:Policy-Claim'] - df['Days:Policy-Accident'])
    
    if 'PastNumberOfClaims' in df.columns:
        if df['PastNumberOfClaims'].dtype == 'object':
            df['PastNumberOfClaims_Numeric'] = df['PastNumberOfClaims'].apply(
                lambda x: 0 if x == 'none' else (
                    1 if x == '1' else (
                        2 if x == '2 to 4' else 5
                    )
                )
            )
        else:
            df['PastNumberOfClaims_Numeric'] = df['PastNumberOfClaims']
        
        df['Multiple_Claims'] = (df['PastNumberOfClaims_Numeric'] > 0).astype(int)
    
    if all(x in df.columns for x in ['PoliceReportFiled', 'WitnessPresent']):
        df['No_Police_No_Witness'] = ((df['PoliceReportFiled'] == 'No') & 
                                     (df['WitnessPresent'] == 'No')).astype(int)
    
    return df


# In[ ]:


# Find the best model with hyperparameter tuning
def tune_best_model(best_model_name, X_train, y_train, models):
    if best_model_name == 'Neural Network':
        print("Skipping hyperparameter tuning for Neural Network as it requires custom implementation")
        return None
    
    best_model = models[best_model_name]
    
    param_grids = {
        'Logistic Regression': {
            'C': [0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        },
        'Decision Tree': {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'Random Forest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        },
        'Gradient Boosting': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        },
        'SVM': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
    }
    
    param_grid = param_grids.get(best_model_name, {})
    
    if not param_grid:
        print(f"No parameter grid defined for {best_model_name}")
        return best_model
    
    print(f"\nPerforming hyperparameter tuning for {best_model_name}...")
    
    grid_search = GridSearchCV(
        estimator=best_model,
        param_grid=param_grid,
        scoring='f1',
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


# In[ ]:


file_path = "carclaims.csv"
df = load_data(file_path)


# In[ ]:


df, preprocessor = preprocess_data(df)


# In[ ]:


perform_eda(df)


# In[ ]:


df = feature_engineering(df)


# In[ ]:


X_train, X_test, y_train, y_test = split_data(df, preprocessor)


# In[ ]:


classical_results, models = train_classical_models(X_train, X_test, y_train, y_test)


# In[ ]:


nn_results, nn_model = train_neural_network(X_train, X_test, y_train, y_test)
    
all_results = classical_results + [nn_results]

comparison = compare_models(all_results)
best_model_name = comparison.loc[comparison['f1'].idxmax()]['model']
print(f"\nBest overall model (by F1 score): {best_model_name}")
    




