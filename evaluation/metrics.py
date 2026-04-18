import numpy as np
from geopy.distance import geodesic

# --- Retrieval Metrics ---

def recall_at_k(query_labels, retrieved_per_query, k_values=[1, 5, 10]):
    recalls = {k: 0 for k in k_values}
    n = len(query_labels)
    for true_lbl, retrieved in zip(query_labels, retrieved_per_query):
        for k in k_values:
            if true_lbl in retrieved[:k]: 
                recalls[k] += 1
    return {k: (v / n) * 100.0 for k, v in recalls.items()}

def average_precision(true_label, retrieved_labels):
    hits, prec = 0, []
    for i, lbl in enumerate(retrieved_labels):
        if lbl == true_label:
            hits += 1
            prec.append(hits / (i + 1))
    return np.mean(prec) if prec else 0.0

def mean_average_precision(query_labels, retrieved_per_query):
    return np.mean([average_precision(q, r) for q, r in zip(query_labels, retrieved_per_query)]) * 100.0

# --- Geo-Localization Metrics ---

def localization_errors(pred_coords, gt_coords):
    # geopy expects (latitude, longitude)
    return [geodesic(pred, gt).meters for pred, gt in zip(pred_coords, gt_coords)]

def pct_within(errors, ds=[1, 5, 25, 100]):
    n = len(errors)
    return {d: sum(e <= d for e in errors) / n * 100.0 for d in ds}