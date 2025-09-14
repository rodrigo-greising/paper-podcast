from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from umap import UMAP
from hdbscan import HDBSCAN

from ..config import Settings
from ..utils.logging import get_logger
from ..utils.paths import run_dir


logger = get_logger(__name__)


def _choose_k(n: int) -> int:
	# Heuristic: sqrt(n/3), clamp 3..20 (kept for legacy KMeans fallback)
	k = max(3, min(20, int(np.sqrt(max(n, 1) / 3))))
	return k


def _evaluate_clustering(X: np.ndarray, labels: np.ndarray) -> dict:
	"""Evaluate clustering quality using multiple metrics."""
	if len(set(labels)) < 2:
		return {}
	
	# Filter out noise points (label -1) for evaluation
	mask = labels != -1
	if sum(mask) < 2:
		return {
			'n_clusters': 0,
			'noise_ratio': sum(~mask) / len(labels)
		}
	
	X_clean = X[mask]
	labels_clean = labels[mask]
	
	return {
		'silhouette_score': silhouette_score(X_clean, labels_clean),
		'davies_bouldin_score': davies_bouldin_score(X_clean, labels_clean),
		'calinski_harabasz_score': calinski_harabasz_score(X_clean, labels_clean),
		'n_clusters': len(set(labels_clean)),
		'noise_ratio': sum(~mask) / len(labels)
	}


def _cluster_hdbscan(X: np.ndarray, min_cluster_size: int = None, is_large_dataset: bool = False) -> tuple[np.ndarray, dict]:
	"""Modern clustering using UMAP + HDBSCAN with optimizations for large datasets."""
	n_samples = X.shape[0]
	
	# Adaptive parameters based on dataset size
	if min_cluster_size is None:
		if is_large_dataset:
			# For large datasets (thousands of papers), create more meaningful topic clusters
			min_cluster_size = max(10, n_samples // 50)  # Smaller clusters for better topic separation
		else:
			min_cluster_size = max(2, n_samples // 20)
	
	# Optimize UMAP parameters for large datasets
	if is_large_dataset:
		n_components = min(15, n_samples // 10) if n_samples > 100 else min(10, n_samples // 2)
		n_neighbors = min(50, n_samples // 5) if n_samples > 100 else min(30, n_samples // 3)
		min_dist = 0.1  # Slightly looser for better separation of distinct topics
	else:
		n_components = min(10, n_samples // 2) if n_samples > 20 else 2
		n_neighbors = min(30, n_samples // 3) if n_samples > 30 else min(15, n_samples - 1)
		min_dist = 0.0
	
	# Apply UMAP for dimensionality reduction
	reducer = UMAP(
		n_components=n_components,
		n_neighbors=n_neighbors,
		min_dist=min_dist,
		random_state=42,
		verbose=is_large_dataset  # Enable verbose logging for large datasets
	)
	X_reduced = reducer.fit_transform(X)
	
	# Apply HDBSCAN clustering with optimized parameters for large datasets
	if is_large_dataset:
		clusterer = HDBSCAN(
			min_cluster_size=min_cluster_size,
			min_samples=max(2, min_cluster_size // 3),  # Require more samples for stability
			metric='euclidean',
			cluster_selection_epsilon=0.1  # Allow some flexibility in cluster boundaries
		)
	else:
		clusterer = HDBSCAN(
			min_cluster_size=min_cluster_size,
			min_samples=1,
			metric='euclidean'
		)
	
	labels = clusterer.fit_predict(X_reduced)
	
	# Evaluate clustering quality
	metrics = _evaluate_clustering(X_reduced, labels)
	
	return labels, metrics


def _label_clusters(texts: List[str], labels: List[int]) -> List[str]:
	vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
	X = vectorizer.fit_transform(texts)
	terms = np.array(vectorizer.get_feature_names_out())

	labels_arr = np.array(labels)
	titles: List[str] = []
	for c in sorted(set(labels)):
		idx = np.where(labels_arr == c)[0]
		centroid = np.asarray(X[idx].mean(axis=0)).ravel()
		top_idx = centroid.argsort()[::-1][:5]
		title = ", ".join(terms[top_idx])
		titles.append(title)
	return titles


def cluster_run(settings: Settings, run_id: str) -> None:
	run_path = run_dir(settings.data_dir, run_id)
	vectors_path = run_path / "vectors.parquet"
	clusters_path = run_path / "clusters.json"

	if not vectors_path.exists():
		raise FileNotFoundError("Missing vectors.parquet; run embed first")

	df = pd.read_parquet(vectors_path)
	texts = df["text"].tolist()
	if not texts:
		logger.warning("No texts to cluster")
		return

	X = np.vstack(df["embedding"].tolist())
	
	# Detect if this is a large dataset (monthly run with thousands of papers)
	is_large_dataset = len(texts) > 1000
	
	# Try modern HDBSCAN clustering first
	labels, metrics = _cluster_hdbscan(X, is_large_dataset=is_large_dataset)
	
	# Fallback to KMeans if HDBSCAN produces too few clusters or too much noise
	if metrics.get('n_clusters', 0) < 2 or metrics.get('noise_ratio', 0) > 0.5:
		logger.info(f"HDBSCAN produced {metrics.get('n_clusters', 0)} clusters with {metrics.get('noise_ratio', 0):.2%} noise. Falling back to KMeans.")
		k = _choose_k(len(texts))
		model = KMeans(n_clusters=k, n_init=10, random_state=42)
		labels = model.fit_predict(X)
		metrics = _evaluate_clustering(X, labels)
		method = "KMeans"
	else:
		method = "HDBSCAN"
	
	# Log clustering results
	logger.info(f"Clustering method: {method}")
	if metrics:
		logger.info(f"Clusters: {metrics.get('n_clusters', 'unknown')}, "
			   f"Silhouette: {metrics.get('silhouette_score', 'N/A'):.3f}, "
			   f"Davies-Bouldin: {metrics.get('davies_bouldin_score', 'N/A'):.3f}")
		if 'noise_ratio' in metrics:
			logger.info(f"Noise ratio: {metrics['noise_ratio']:.2%}")

	df["cluster"] = labels

	# Aggregate to cluster-level
	cluster_to_texts = {c: [] for c in sorted(set(labels))}
	cluster_to_papers = {c: set() for c in sorted(set(labels))}
	for row in df.itertuples(index=False):
		cluster_to_texts[row.cluster].append(row.text)
		cluster_to_papers[row.cluster].add(row.paper_id)

	titles = _label_clusters(
		texts=["\n".join(cluster_to_texts[c]) for c in sorted(cluster_to_texts.keys())],
		labels=list(sorted(cluster_to_texts.keys())),
	)

	clusters = []
	for i, c in enumerate(sorted(cluster_to_texts.keys())):
		clusters.append({
			"cluster_id": int(c),
			"label": titles[i],
			"paper_ids": sorted(list(cluster_to_papers[c])),
		})

	with open(clusters_path, "w", encoding="utf-8") as f:
		json.dump(clusters, f, ensure_ascii=False, indent=2)
	
	# Save clustering metadata
	cluster_metadata = {
		"method": method,
		"metrics": metrics,
		"clusters": clusters
	}
	
	with open(clusters_path.with_suffix('.metadata.json'), "w", encoding="utf-8") as f:
		json.dump(cluster_metadata, f, ensure_ascii=False, indent=2)

	logger.info(f"Wrote clusters to {clusters_path}")
	logger.info(f"Wrote metadata to {clusters_path.with_suffix('.metadata.json')}")
