import pickle
import pandas as pd
import os

MODEL_PATH = 'ids_model.pkl'
TRAIN_CSV = 'Train_data.csv'

def _find_class_label(enc_le, requested):
    # match ignoring case to one of the encoder classes
    classes = list(enc_le.classes_)
    for c in classes:
        if str(c).lower() == str(requested).lower():
            return c
    # no exact match; try substring match
    for c in classes:
        if str(requested).lower() in str(c).lower() or str(c).lower() in str(requested).lower():
            return c
    raise ValueError(f"Requested class '{requested}' not found in model classes: {classes}")

def main(requested_type: str):
    """Return (pred_idx, prob_list, predsvm, probsvm, classes_list).

    Uses the saved `ids_model.pkl` and a representative sample from `Train_data.csv`
    for the requested class to produce probabilities. predsvm/probsvm mirror the
    primary model for compatibility with the example Flask app.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    with open(MODEL_PATH,'rb') as f:
        data = pickle.load(f)

    model = data['model']
    selected = data.get('selected', [])
    scaler = data.get('scaler', None)
    encoders = data.get('encoders', {})

    # determine class label name and numeric code
    if 'class' in encoders:
        class_le = encoders['class']
        matched = _find_class_label(class_le, requested_type)
        class_code = int(class_le.transform([matched])[0])
        classes = list(class_le.classes_)
    else:
        # fallback: load train csv and inspect unique values
        if not os.path.exists(TRAIN_CSV):
            raise FileNotFoundError('No class encoder and train CSV not found to infer classes')
        train = pd.read_csv(TRAIN_CSV)
        uniq = sorted(train['class'].unique().tolist())
        # try to match
        for c in uniq:
            if str(c).lower() == str(requested_type).lower():
                class_code = c
                break
        else:
            raise ValueError(f"Requested class '{requested_type}' not found in training data classes: {uniq}")
        classes = uniq

    # load train CSV to extract a representative sample row for that class
    if not os.path.exists(TRAIN_CSV):
        raise FileNotFoundError(f"Training CSV not found: {TRAIN_CSV}")

    train = pd.read_csv(TRAIN_CSV)

    # find a row matching that class (string match against original 'class' column)
    # Note: train['class'] contains original labels (strings) in this dataset
    if 'class' in encoders:
        target_str = matched
    else:
        target_str = class_code

    subset = train[train['class'].astype(str).str.lower() == str(target_str).lower()]
    if subset.shape[0] == 0:
        # fallback to any row
        row = train.iloc[0]
    else:
        row = subset.iloc[0]

    # build input row for selected features
    input_vals = []
    for col in selected:
        if col not in train.columns:
            input_vals.append(0)
            continue
        val = row[col]
        # if this column had an encoder, map using it (safe mapping)
        if col in encoders:
            le = encoders[col]
            cls = list(le.classes_)
            mapping = {v: i for i, v in enumerate(cls)}
            mapped = mapping.get(str(val), -1)
            input_vals.append(mapped)
        else:
            # numeric expected
            try:
                input_vals.append(float(val))
            except Exception:
                input_vals.append(0.0)

    X = pd.DataFrame([input_vals], columns=selected)
    if scaler is not None:
        Xs = scaler.transform(X)
    else:
        Xs = X.values

    # predict probabilities; some estimators may not have predict_proba
    try:
        probs = model.predict_proba(Xs)[0].tolist()
    except Exception:
        # fallback: get decision_function or predict
        try:
            preds = model.decision_function(Xs)
            # squash to [0,1]
            import numpy as _np
            s = _np.exp(preds - _np.max(preds))
            probs = (s / _np.sum(s)).ravel().tolist()
        except Exception:
            preds = model.predict(Xs)
            # make one-hot
            probs = [0.0] * len(classes)
            if len(preds) > 0:
                # find index of predicted class in classes (if encoded, map)
                p = preds[0]
                try:
                    idx = list(model.classes_).index(p)
                except Exception:
                    idx = 0
                probs[idx] = 1.0

    # map probs to classes order (model.classes_ corresponds to encoded labels)
    # If encoders['class'] exists, its classes_ provide the human-readable names in order
    if 'class' in encoders:
        classes = list(encoders['class'].classes_)
    else:
        # try to use model.classes_
        classes = [str(c) for c in getattr(model, 'classes_', [])]

    # get predicted index (highest probability)
    try:
        pred_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
    except Exception:
        pred_idx = 0

    # For compatibility return also 'svm' placeholders (same values)
    predsvm = pred_idx
    probsvm = probs

    return pred_idx, probs, predsvm, probsvm, classes

if __name__ == '__main__':
    print(main('normal'))
