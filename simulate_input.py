#!/usr/bin/env python3
"""
simulate_input.py

Create synthetic or user-provided traffic samples compatible with the saved
`ids_model.pkl` and run the model to get predictions + probabilities.

Usage examples:
  - Generate one interactive sample:
      python simulate_input.py
  - Provide a single sample via key=value pairs:
      python simulate_input.py --sample "protocol_type=tcp,flag=SF,src_bytes=181,dst_bytes=545"
  - Generate 5 random samples:
      python simulate_input.py --random 5
  - Classify a CSV of samples (columns may include selected feature names):
      python simulate_input.py --csv input.csv --out results.csv

The script will map categorical values using encoders saved in the model
and will fill missing features with reasonable defaults (encoder first-class
or scaler mean when available).
"""
import argparse
import os
import pickle
import random
import sys
from typing import Dict, Any

import numpy as np
import pandas as pd


MODEL_PATH = 'ids_model.pkl'


def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def parse_kv_list(s: str) -> Dict[str, str]:
    out = {}
    for part in s.split(','):
        if '=' not in part:
            continue
        k, v = part.split('=', 1)
        out[k.strip()] = v.strip()
    return out


def build_row_from_inputs(selected, encoders, scaler, inputs: Dict[str, Any]):
    vals = []
    for i, col in enumerate(selected):
        if col in inputs and inputs[col] is not None:
            raw = inputs[col]
            # if encoder exists treat as categorical mapping
            if col in encoders:
                le = encoders[col]
                mapping = {str(v): idx for idx, v in enumerate(list(le.classes_))}
                mapped = mapping.get(str(raw), -1)
                vals.append(int(mapped))
            else:
                try:
                    vals.append(float(raw))
                except Exception:
                    # fallback to numeric default
                    if scaler is not None:
                        vals.append(float(scaler.mean_[i]))
                    else:
                        vals.append(0.0)
        else:
            # not provided -> choose sensible default
            if col in encoders:
                le = encoders[col]
                # pick the first known class as default
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
                # sample around the mean
                inputs[col] = max(0.0, random.gauss(mean, max(1.0, scale)))
            else:
                inputs[col] = float(random.uniform(0, 100))
    return inputs


def predict_df(model, scaler, encoders, df, selected):
    # df is expected to already contain selected columns but may be raw values
    X = df.reindex(columns=selected, fill_value=0)

    # apply encoders for any categorical columns present in encoders
    for c in selected:
        if c in encoders and c in X.columns:
            le = encoders[c]
            mapping = {str(v): i for i, v in enumerate(list(le.classes_))}
            X[c] = X[c].astype(str).map(lambda v: mapping.get(v, -1)).astype(int)
        else:
            # ensure numeric
            try:
                X[c] = pd.to_numeric(X[c])
            except Exception:
                X[c] = 0

    if scaler is not None:
        Xs = scaler.transform(X)
    else:
        Xs = X.values

    # get probabilities if available
    try:
        probs = model.predict_proba(Xs)
    except Exception:
        probs = None

    try:
        preds = model.predict(Xs)
    except Exception:
        preds = None

    return preds, probs


def main():
    parser = argparse.ArgumentParser(description='Simulate traffic inputs and run IDS model predictions')
    parser.add_argument('--sample', help='Single sample as comma-separated key=value pairs')
    parser.add_argument('--random', type=int, help='Generate N random samples')
    parser.add_argument('--csv', help='CSV file with sample rows to classify')
    parser.add_argument('--out', help='Optional output CSV to save predictions')
    args = parser.parse_args()

    data = load_model()
    model = data['model']
    selected = list(data.get('selected', []))
    scaler = data.get('scaler', None)
    encoders = data.get('encoders', {}) or {}

    if len(selected) == 0:
        print('Model has no `selected` features recorded. Exiting.')
        sys.exit(1)

    rows = []

    if args.csv:
        df = pd.read_csv(args.csv)
        # attempt to map/convert columns to selected features
        # If CSV contains human-readable categorical values, mapping happens in predict_df
        preds, probs = predict_df(model, scaler, encoders, df, selected)
        out_df = df.copy()
        if preds is not None:
            out_df['prediction'] = preds
        if probs is not None:
            # expand prob columns
            for i in range(probs.shape[1]):
                out_df[f'prob_{i}'] = probs[:, i]
        print(out_df.head())
        if args.out:
            out_df.to_csv(args.out, index=False)
            print(f'Wrote results to {args.out}')
        return

    if args.random:
        N = args.random
        results = []
        for _ in range(N):
            inp = random_sample(selected, encoders, scaler)
            df_row = build_row_from_inputs(selected, encoders, scaler, inp)
            preds, probs = predict_df(model, scaler, encoders, df_row, selected)
            human_pred = None
            prob_list = None
            if preds is not None:
                human_pred = preds[0]
            if probs is not None:
                prob_list = probs[0].tolist()
            results.append({'input': inp, 'prediction': human_pred, 'probs': prob_list})
        for r in results:
            print(r)
        return

    if args.sample:
        inputs = parse_kv_list(args.sample)
        df_row = build_row_from_inputs(selected, encoders, scaler, inputs)
        preds, probs = predict_df(model, scaler, encoders, df_row, selected)
        if preds is not None:
            pred = int(preds[0])
        else:
            pred = None
        if probs is not None:
            probs_list = probs[0].tolist()
        else:
            probs_list = None
        # If class encoder exists, map numeric predictions back to readable labels
        if 'class' in encoders and pred is not None:
            class_le = encoders['class']
            try:
                human_label = class_le.classes_[pred]
            except Exception:
                human_label = str(pred)
        else:
            human_label = str(pred)

        print('Input (as model-ready row):')
        print(df_row.to_string(index=False))
        print('\nPrediction:', human_label)
        print('Probabilities:', probs_list)
        return

    # interactive path: prompt for each selected feature
    print('No input flags provided — entering interactive mode.')
    print('Selected (model) features order:')
    for c in selected:
        print(' -', c)
    user_vals = {}
    for c in selected:
        v = input(f'Value for {c} (leave blank for default): ').strip()
        user_vals[c] = v if v != '' else None

    df_row = build_row_from_inputs(selected, encoders, scaler, user_vals)
    preds, probs = predict_df(model, scaler, encoders, df_row, selected)
    if preds is not None:
        pred = int(preds[0])
    else:
        pred = None
    probs_list = probs[0].tolist() if probs is not None else None
    if 'class' in encoders and pred is not None:
        class_le = encoders['class']
        try:
            human_label = class_le.classes_[pred]
        except Exception:
            human_label = str(pred)
    else:
        human_label = str(pred)

    print('\nInput (as model-ready row):')
    print(df_row.to_string(index=False))
    print('\nPrediction:', human_label)
    print('Probabilities:', probs_list)


if __name__ == '__main__':
    main()
