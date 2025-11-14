import streamlit as st
import pandas as pd
import numpy as np
import io
import pickle
import os
import random
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


##### Simulator helper functions (merged from simulate_input.py) #####
def parse_kv_list(s: str):
    out = {}
    for part in s.split(','):
        if '=' not in part:
            continue
        k, v = part.split('=', 1)
        out[k.strip()] = v.strip()
    return out


def build_row_from_inputs(selected, encoders, scaler, inputs: dict):
    vals = []
    for i, col in enumerate(selected):
        if col in inputs and inputs[col] is not None:
            raw = inputs[col]
            if col in encoders:
                le = encoders[col]
                mapping = {str(v): idx for idx, v in enumerate(list(le.classes_))}
                mapped = mapping.get(str(raw), -1)
                vals.append(int(mapped))
            else:
                try:
                    vals.append(float(raw))
                except Exception:
                    if scaler is not None:
                        vals.append(float(scaler.mean_[i]))
                    else:
                        vals.append(0.0)
        else:
            if col in encoders:
                le = encoders[col]
                default = le.classes_[0] if len(le.classes_) > 0 else -1
                mapping = {str(v): idx for idx, v in enumerate(list(le.classes_))}
                vals.append(int(mapping.get(str(default), -1)))
            else:
                if scaler is not None:
                    vals.append(float(scaler.mean_[i]))
                else:
                    vals.append(0.0)

    return pd.DataFrame([vals], columns=selected)


def random_sample(selected, encoders, scaler):
    inputs = {}
    for i, col in enumerate(selected):
        if col in encoders:
            le = encoders[col]
            if len(le.classes_) > 0:
                inputs[col] = random.choice(list(le.classes_))
            else:
                inputs[col] = ''
        else:
            if scaler is not None:
                mean = float(scaler.mean_[i])
                scale = float(getattr(scaler, 'scale_', np.ones_like(scaler.mean_))[i])
                inputs[col] = max(0.0, random.gauss(mean, max(1.0, scale)))
            else:
                inputs[col] = float(random.uniform(0, 100))
    return inputs


def predict_df(model, scaler, encoders, df, selected):
    X = df.reindex(columns=selected, fill_value=0)
    for c in selected:
        if c in encoders and c in X.columns:
            le = encoders[c]
            mapping = {str(v): i for i, v in enumerate(list(le.classes_))}
            X[c] = X[c].astype(str).map(lambda v: mapping.get(v, -1)).astype(int)
        else:
            try:
                X[c] = pd.to_numeric(X[c])
            except Exception:
                X[c] = 0

    if scaler is not None:
        Xs = scaler.transform(X)
    else:
        Xs = X.values

    try:
        probs = model.predict_proba(Xs)
    except Exception:
        probs = None

    try:
        preds = model.predict(Xs)
    except Exception:
        preds = None

    return preds, probs


def _pred_to_human(pred, encoders, model):
    """Map numeric prediction to a simple human-friendly status.

    Returns (label, status) where label is the original class label if available
    (e.g. 'normal'/'anomaly') and status is either 'Normal' or 'Attack'.
    """
    if pred is None:
        return None, 'Unknown'
    # pred may be scalar or array-like; ensure scalar
    try:
        p = int(pred)
    except Exception:
        # if model returns non-integer classes, try to map directly
        p = pred

    label = None
    if 'class' in encoders:
        try:
            label = encoders['class'].classes_[p]
        except Exception:
            label = str(p)
    else:
        try:
            # fall back to model.classes_
            label = str(list(getattr(model, 'classes_', []))[int(p)])
        except Exception:
            label = str(p)

    # normalize to Attack vs Normal
    if isinstance(label, str) and label.lower() == 'normal':
        status = 'Normal'
    else:
        status = 'Attack'
    return label, status

##### end simulator helpers #####
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
                    # Map numeric preds to human-friendly status and show only that
                    human_labels = []
                    statuses = []
                    for p in preds:
                        lab, stat = _pred_to_human(p, pretrained.get('encoders', {}), model)
                        human_labels.append(lab)
                        statuses.append(stat)
                    out = df.copy()
                    out['label'] = human_labels
                    out['status'] = statuses
                    st.success('Classification complete')
                    st.write(out[['label','status']].head(50))
                    st.download_button('Download classification (CSV)', data=out.to_csv(index=False).encode('utf-8'), file_name='classification.csv')
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
                            lab, stat = _pred_to_human(pred, pretrained.get('encoders', {}), model)
                            if stat == 'Attack':
                                st.error('Intrusion detected')
                            else:
                                st.success('No intrusion detected')
                except Exception as e:
                    st.exception(e)
            # Simulator: generate or craft model-compatible inputs and classify
            with st.expander('Simulator (generate model-compatible inputs)'):
                st.write('Create synthetic or custom samples using the model\'s encoders and selected features.')
                # Manual friendly-input simulator
                with st.expander('Manual Simulation (friendly inputs)'):
                    st.write('Enter connection-level values and map them to the model features.')
                    duration = st.number_input('Duration (0-54451)', min_value=0, max_value=54451, value=0)
                    # protocol options from encoder if present
                    encoders = pretrained.get('encoders', {}) or {}
                    if 'protocol_type' in encoders:
                        proto_opts = list(encoders['protocol_type'].classes_)
                        protocol_type = st.selectbox('Protocol Type', options=proto_opts)
                    else:
                        protocol_type = st.selectbox('Protocol Type', options=['tcp','udp','icmp'])

                    if 'service' in encoders:
                        svc_opts = list(encoders['service'].classes_)
                        service = st.selectbox('Service', options=svc_opts)
                    else:
                        service = st.text_input('Service (e.g. http, ftp)', value='http')

                    if 'flag' in encoders:
                        flag_opts = list(encoders['flag'].classes_)
                        flag = st.selectbox('Flag', options=flag_opts)
                    else:
                        flag = st.text_input('Flag (e.g. SF, S0)', value='SF')

                    src_bytes = st.number_input('Src Bytes', min_value=0, value=0)
                    dst_bytes = st.number_input('Dstn Bytes', min_value=0, value=0)
                    logged_in = st.selectbox('Logged In', options=[0,1], index=0)
                    wrong_fragment = st.number_input('Wrong Fragment', min_value=0, value=0)
                    same_destn_count = st.number_input('Same Destn Count', min_value=0, value=0)
                    same_port_count = st.number_input('Same Port Count', min_value=0, value=0)

                    if st.button('Simulate connection'):
                        try:
                            selected = pretrained.get('selected', [])
                            scaler = pretrained.get('scaler', None)
                            encs = pretrained.get('encoders', {}) or {}

                            # map friendly inputs to model-selected features
                            mapped_inputs = {}
                            for col in selected:
                                if col == 'protocol_type':
                                    mapped_inputs[col] = protocol_type
                                elif col == 'flag':
                                    mapped_inputs[col] = flag
                                elif col == 'src_bytes':
                                    mapped_inputs[col] = src_bytes
                                elif col == 'dst_bytes':
                                    mapped_inputs[col] = dst_bytes
                                elif col == 'count':
                                    mapped_inputs[col] = same_port_count
                                elif col == 'same_srv_rate':
                                    mapped_inputs[col] = float(logged_in)
                                elif col == 'diff_srv_rate':
                                    # small normalization for wrong_fragment
                                    mapped_inputs[col] = min(1.0, float(wrong_fragment) / 10.0)
                                elif col == 'dst_host_srv_count':
                                    mapped_inputs[col] = same_destn_count
                                elif col == 'dst_host_same_srv_rate':
                                    mapped_inputs[col] = float(logged_in)
                                elif col == 'dst_host_diff_srv_rate':
                                    mapped_inputs[col] = min(1.0, float(wrong_fragment) / 10.0)
                                else:
                                    # unknown selected column -> leave None so builder fills default
                                    mapped_inputs[col] = None

                            row = build_row_from_inputs(selected, encs, scaler, mapped_inputs)
                            preds, probs = predict_df(pretrained['model'], scaler, encs, row, selected)
                            p = int(preds[0]) if preds is not None else None
                            lab, stat = _pred_to_human(p, encs, pretrained['model'])
                            if stat == 'Attack':
                                st.error('Intrusion detected')
                            else:
                                st.success('No intrusion detected')
                        except Exception as e:
                            st.exception(e)
                sim_mode = st.selectbox('Simulator mode', ['Key=Value sample', 'Random samples', 'Upload CSV for simulation'])
                if sim_mode == 'Key=Value sample':
                    kv = st.text_input('Enter comma-separated key=value pairs (e.g. protocol_type=tcp,src_bytes=100)')
                    if st.button('Classify KV sample'):
                        try:
                            if not kv:
                                st.error('Provide key=value pairs')
                            else:
                                inputs = parse_kv_list(kv)
                                selected = pretrained.get('selected', [])
                                scaler = pretrained.get('scaler', None)
                                encoders = pretrained.get('encoders', {}) or {}
                                row = build_row_from_inputs(selected, encoders, scaler, inputs)
                                preds, probs = predict_df(pretrained['model'], scaler, encoders, row, selected)
                                pred = int(preds[0]) if preds is not None else None
                                lab, stat = _pred_to_human(pred, pretrained.get('encoders', {}), pretrained['model'])
                                st.write('Model-ready input:')
                                st.write(row)
                                if stat == 'Attack':
                                    st.error('Intrusion detected')
                                else:
                                    st.success('No intrusion detected')
                        except Exception as e:
                            st.exception(e)
                elif sim_mode == 'Random samples':
                    n = st.number_input('Number of random samples', min_value=1, max_value=100, value=3)
                    if st.button('Generate random samples'):
                        try:
                            selected = pretrained.get('selected', [])
                            scaler = pretrained.get('scaler', None)
                            encoders = pretrained.get('encoders', {}) or {}
                            rows = []
                            for _ in range(int(n)):
                                inp = random_sample(selected, encoders, scaler)
                                row = build_row_from_inputs(selected, encoders, scaler, inp)
                                preds, probs = predict_df(pretrained['model'], scaler, encoders, row, selected)
                                p = int(preds[0]) if preds is not None else None
                                lab, stat = _pred_to_human(p, encoders, pretrained['model'])
                                rows.append({**inp, 'label': lab, 'status': stat})
                            st.write(pd.DataFrame(rows))
                        except Exception as e:
                            st.exception(e)
                else:
                    up = st.file_uploader('Upload CSV to simulate (columns may include selected features)', type='csv')
                    if up is not None:
                        try:
                            df = pd.read_csv(up)
                            preds, probs = predict_df(pretrained['model'], pretrained.get('scaler', None), pretrained.get('encoders', {}), df, pretrained.get('selected', []))
                            out_df = df.copy()
                            if preds is not None:
                                out_df['prediction'] = preds
                            if probs is not None:
                                for i in range(probs.shape[1]):
                                    out_df[f'prob_{i}'] = probs[:, i]
                            st.write(out_df.head())
                            st.download_button('Download simulation results', data=out_df.to_csv(index=False).encode('utf-8'), file_name='simulation_results.csv')
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
