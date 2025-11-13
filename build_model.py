import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

# Simple build script to create ids_model.pkl with model, selected features, scaler and encoders
TRAIN = 'Train_data.csv'
TEST = 'Test_data.csv'

train = pd.read_csv(TRAIN)
test = pd.read_csv(TEST)

# build encoders for object columns
encoders = {}
for col in train.columns:
    if train[col].dtype == 'object':
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col].astype(str))
        encoders[col] = le
for col in test.columns:
    if test[col].dtype == 'object':
        if col in encoders:
            # transform test values safely: unseen labels -> -1
            le = encoders[col]
            cls = list(le.classes_)
            mapping = {v: i for i, v in enumerate(cls)}
            test[col] = test[col].astype(str).map(lambda v: mapping.get(v, -1)).astype(int)
        else:
            le = LabelEncoder(); test[col] = le.fit_transform(test[col].astype(str)); encoders[col] = le

if 'class' not in train.columns:
    raise SystemExit("'class' column not found in training data")

X = train.drop(['class'], axis=1)
y = train['class']

selector = RFE(RandomForestClassifier(n_estimators=50, random_state=0), n_features_to_select=min(10, X.shape[1]))
selector = selector.fit(X, y)
selected = list(X.columns[selector.get_support()])

X_sel = X[selected]
test_sel = test.reindex(columns=selected, fill_value=0)

scaler = StandardScaler()
Xs = scaler.fit_transform(X_sel)

model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(Xs, y)

out = {'model': model, 'selected': selected, 'scaler': scaler, 'encoders': encoders}
with open('ids_model.pkl', 'wb') as f:
    pickle.dump(out, f)

print('Saved ids_model.pkl with model, selected features, scaler and encoders')