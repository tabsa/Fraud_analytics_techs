"""
Example of sklearn pipeline for a simple ML model, so you can debug what happens in each part when building a pipeline
"""

#%% Import packages
from pathlib import Path
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, QuantileTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
import numpy as np
from sklearn.metrics import accuracy_score
# ML methods from sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

#%% Constants and parameters
LOG = LogisticRegression(solver='liblinear')  
SVM = SVC()

#%% Main script
if __name__ == '__main__':
    # Read pickle file
    wok_dir = Path.cwd() / 'datasets/'
    filename = 'facies_dataset_preprocess.parquet'
    path_file = wok_dir / filename
    data = pd.read_parquet(path_file) # Read file
    # X, Y datasets
    X = data[['Depth', 'GR', 'ILD_log10','DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']]
    y = data['Facies']

    ## Build baseline model (without feature extraction or similar)
    # model = LOG
    # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # # evaluate model
    # scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    # print('Accuracy: %.3f' % (np.mean(scores)))

    ## Build baseline model (with feat eng as a pipeline)
    # Features transforms
    transforms = list()
    transforms.append(('sca', StandardScaler()))
    transforms.append(('qt', QuantileTransformer(n_quantiles=100, output_distribution='normal')))
    # transforms.append(('kbd', KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')))
    # transforms.append(('pca', PCA(n_components=7)))
    # transforms.append(('svd', TruncatedSVD(n_components=7)))

    # Feature union
    fu = FeatureUnion(transforms)
    # Define feature selection
    # rfe = RFE(estimator=LogisticRegression(solver='liblinear'), n_features_to_select=30)

    # Model selection
    model = LOG
    #-------------------------------------------------- use pipeline to chain operation
    steps = list()
    steps.append(('fu', fu))
    # steps.append(('rfe', rfe))
    steps.append(('ml', model))
    pipeline = Pipeline(steps=steps)
    # define the cross-validation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate model
    scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    # report performance
    print('Accuracy: %.3f' % (np.mean(scores)))
