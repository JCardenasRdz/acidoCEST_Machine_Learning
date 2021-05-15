import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: -0.22191660069271624
exported_pipeline = make_pipeline(
    FastICA(tol=0.55),
    VarianceThreshold(threshold=0.0001),
    ElasticNetCV(l1_ratio=0.1, tol=0.01)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
