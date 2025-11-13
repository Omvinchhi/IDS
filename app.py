import streamlit as st
import pandas as pd
import numpy as np
import io
import pickle
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None


st.set_page_config(page_title="IDS Model Playground", layout="wide")

st.title("IDS — Model Playground")
st.markdown(
    "Upload or use provided dataset, train models, and inspect results. This is a lightweight Streamlit UI for the IDS project." 
)

@st.cache_data
def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test

def label_encode_df(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    return df

def preprocess(train, test, drop_cols=None, n_features=10):
    if drop_cols is None:
        drop_cols = []
    train = train.copy()
    test = test.copy()
    for c in drop_cols:
        if c in train.columns:
            train.drop(c, axis=1, inplace=True)
        if c in test.columns:
            test.drop(c, axis=1, inplace=True)

    train = label_encode_df(train)
    test = label_encode_df(test)

    if 'class' not in train.columns:
        raise ValueError("'class' column not found in training data")

    X = train.drop(['class'], axis=1)
    y = train['class']

    # feature selection using RFE with RandomForest
    selector = RFE(RandomForestClassifier(n_estimators=20, random_state=0), n_features_to_select=min(n_features, X.shape[1]))
    selector = selector.fit(X, y)
    selected = X.columns[selector.get_support()]

    X_sel = X[selected]
    test_sel = test.reindex(columns=selected, fill_value=0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sel)
    test_scaled = scaler.transform(test_sel)

    return X_scaled, y.values, test_scaled, selected, scaler

def get_model(name, params):
    if name == 'KNN':
        return KNeighborsClassifier(n_neighbors=int(params.get('n_neighbors', 5)))
    if name == 'LogisticRegression':
        return LogisticRegression(max_iter=1000)
    if name == 'DecisionTree':
        return DecisionTreeClassifier(max_depth=params.get('max_depth', None))
    if name == 'RandomForest':
        return RandomForestClassifier(n_estimators=int(params.get('n_estimators', 100)), random_state=0)
    if name == 'XGBoost':
        if XGBClassifier is None:
            raise ImportError('xgboost is not available in this environment')
        return XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=0)
    if name == 'LightGBM':
        if LGBMClassifier is None:
            raise ImportError('lightgbm is not available in this environment')
        return LGBMClassifier(random_state=0)
    raise ValueError(f'Unknown model: {name}')

def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    return {
        'model': model,
        'train_acc': accuracy_score(y_train, y_pred_train),
        'test_acc': accuracy_score(y_test, y_pred_test),
        'report': classification_report(y_test, y_pred_test, output_dict=False),
        'confusion': confusion_matrix(y_test, y_pred_test)
    }
with st.sidebar:
    st.header('Mode')
    mode = st.radio('Select mode', ['Detection', 'Train (Advanced)'])
    st.markdown('---')
    use_upload = st.checkbox('Upload CSVs for training (advanced)', value=False)
    if use_upload and mode == 'Train (Advanced)':
        train_file = st.file_uploader('Train CSV', type='csv')
        test_file = st.file_uploader('Test CSV', type='csv')
    else:
        train_file = None
        test_file = None

    # detection settings
    detection_input_type = st.selectbox('Detection input', ['Upload CSV of samples', 'Enter single sample (comma-separated values)'])

    # training settings (advanced)
    n_features = st.number_input('Number of features to select (RFE)', min_value=3, max_value=100, value=10, key='nfeat')
    test_size = st.slider('Test set proportion', 0.1, 0.5, 0.3, key='tsize')
    random_state = st.number_input('Random state', value=2, key='rstate')
    model_name = st.selectbox('Model (advanced)', options=['RandomForest','KNN','LogisticRegression','DecisionTree','XGBoost','LightGBM'])
    if model_name == 'KNN':
        knn_k = st.number_input('K for KNN', min_value=1, max_value=50, value=5)
    drop_cols_input = st.text_input('Columns to drop (comma separated)', value='num_outbound_cmds', key='dropcols')

# Try to load pretrained model if available
MODEL_PATH = 'ids_model.pkl'
pretrained = None
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH,'rb') as f:
            pretrained = pickle.load(f)
            st.sidebar.success('Loaded pretrained model')
    except Exception as e:
        st.sidebar.error(f'Failed loading pretrained model: {e}')

if mode == 'Detection':
    st.header('Intrusion Detection')
    if pretrained is None:
        st.warning('No pretrained model found. Use Train (Advanced) to create one, or place `ids_model.pkl` in the repo root.')
    else:
        st.write('Model ready. You can upload CSV(s) of samples or enter a single sample row.')
        if detection_input_type == 'Upload CSV of samples':
            uploaded = st.file_uploader('Upload CSV to classify', type='csv')
            if uploaded is not None:
                try:
                    df = pd.read_csv(uploaded)
                    encoders = pretrained.get('encoders', {}) or {}
                    # apply label encoders where available, else try simple conversion
                    def safe_transform_series(series, encoder):
                        # encoder: LabelEncoder fitted on training values
                        cls = list(encoder.classes_)
                        mapping = {v: i for i, v in enumerate(cls)}
                        return series.astype(str).map(lambda v: mapping.get(v, -1)).astype(int)

                    for c in df.columns:
                        if df[c].dtype == 'object':
                            if c in encoders:
                                df[c] = safe_transform_series(df[c], encoders[c])
                            else:
                                le = LabelEncoder()
                                df[c] = le.fit_transform(df[c].astype(str))

                    selected = pretrained.get('selected', [])
                    scaler = pretrained.get('scaler', None)
                    model = pretrained['model']
                    X = df.reindex(columns=selected, fill_value=0)
                    if scaler is not None:
                        Xs = scaler.transform(X)
                    else:
                        Xs = X.values
                    preds = model.predict(Xs)
                    df['prediction'] = preds
                    st.success('Predictions complete')
                    st.write(df.head(50))
                    st.download_button('Download predictions (CSV)', data=df.to_csv(index=False).encode('utf-8'), file_name='predictions.csv')
                except Exception as e:
                    st.exception(e)
        else:
            sample = st.text_input('Enter comma-separated values matching selected features order')
            if st.button('Classify sample'):
                try:
                    selected = pretrained.get('selected', [])
                    scaler = pretrained.get('scaler', None)
                    model = pretrained['model']
                    if not sample:
                        st.error('Please provide sample values')
                    else:
                        vals = [v.strip() for v in sample.split(',')]
                        if len(vals) != len(selected):
                            st.error(f'Expected {len(selected)} values (selected features)')
                        else:
                            row = pd.DataFrame([vals], columns=selected)
                            # attempt to convert numeric columns
                            def safe_transform_val(val, encoder):
                                cls = list(encoder.classes_)
                                mapping = {v: i for i, v in enumerate(cls)}
                                return mapping.get(str(val), -1)

                            for c in row.columns:
                                try:
                                    row[c] = pd.to_numeric(row[c])
                                except Exception:
                                    # leave as string -> encode or map unseen -> -1
                                    if 'encoders' in pretrained and c in pretrained['encoders']:
                                        row[c] = safe_transform_val(row[c].iloc[0], pretrained['encoders'][c])
                                    else:
                                        le = LabelEncoder(); row[c] = le.fit_transform(row[c].astype(str))
                            Xs = scaler.transform(row) if scaler is not None else row.values
                            pred = model.predict(Xs)[0]
                            st.metric('Prediction', str(pred))
                except Exception as e:
                    st.exception(e)
elif mode == 'Train (Advanced)':
    st.header('Train model (advanced)')
    if st.button('Load & Preprocess (advanced)'):
        try:
            if use_upload and (train_file is not None and test_file is not None):
                train = pd.read_csv(train_file)
                test = pd.read_csv(test_file)
            else:
                train, test = load_data('Train_data.csv', 'Test_data.csv')
            drop_cols = [c.strip() for c in drop_cols_input.split(',') if c.strip()]
            # build encoders map and preprocess here (reuse earlier preprocess but also return encoders)
            # label encode with encoders stored
            encoders = {}
            for col in train.columns:
                if train[col].dtype == 'object':
                    le = LabelEncoder(); train[col] = le.fit_transform(train[col].astype(str)); encoders[col] = le
            for col in test.columns:
                if test[col].dtype == 'object':
                    if col in encoders:
                        test[col] = encoders[col].transform(test[col].astype(str))
                    else:
                        le = LabelEncoder(); test[col] = le.fit_transform(test[col].astype(str)); encoders[col] = le

            if 'class' not in train.columns:
                st.error("'class' column not found in training data")
            else:
                X = train.drop(['class'] + drop_cols, axis=1, errors='ignore')
                y = train['class']
                selector = RFE(RandomForestClassifier(n_estimators=50, random_state=0), n_features_to_select=min(n_features, X.shape[1]))
                selector = selector.fit(X, y)
                selected = list(X.columns[selector.get_support()])
                X_sel = X[selected]
                test_sel = test.reindex(columns=selected, fill_value=0)
                scaler = StandardScaler(); Xs = scaler.fit_transform(X_sel); testXs = scaler.transform(test_sel)
                st.session_state['train_ready'] = True
                st.session_state['train_X'] = Xs; st.session_state['train_y'] = y.values; st.session_state['test_X'] = testXs; st.session_state['selected'] = selected; st.session_state['scaler'] = scaler; st.session_state['encoders'] = encoders
                st.success('Preprocessing done (advanced). Click Train model to continue.')
        except Exception as e:
            st.exception(e)
    if 'train_ready' in st.session_state and st.button('Train model (advanced)'):
        try:
            Xs = st.session_state['train_X']; ys = st.session_state['train_y']
            if model_name == 'RandomForest':
                model = RandomForestClassifier(n_estimators=100, random_state=0)
            elif model_name == 'KNN':
                model = KNeighborsClassifier(n_neighbors=int(knn_k))
            elif model_name == 'LogisticRegression':
                model = LogisticRegression(max_iter=1000)
            else:
                model = RandomForestClassifier(n_estimators=100, random_state=0)
            model.fit(Xs, ys)
            # save model with metadata
            out = {'model': model, 'selected': st.session_state['selected'], 'scaler': st.session_state['scaler'], 'encoders': st.session_state.get('encoders', {})}
            with open(MODEL_PATH,'wb') as f:
                pickle.dump(out, f)
            st.success(f'Model trained and saved to {MODEL_PATH}')
        except Exception as e:
            st.exception(e)

