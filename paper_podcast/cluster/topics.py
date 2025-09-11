from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from ..config import Settings
from ..utils.logging import get_logger
from ..utils.paths import run_dir


logger = get_logger(__name__)


def _choose_k(n: int) -> int:
	# Heuristic: sqrt(n/3), clamp 3..20
	k = max(3, min(20, int(np.sqrt(max(n, 1) / 3))))
	return k


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

	k = _choose_k(len(texts))
	X = np.vstack(df["embedding"].tolist())
	model = KMeans(n_clusters=k, n_init=10, random_state=42)
	labels = model.fit_predict(X)

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

	logger.info(f"Wrote clusters to {clusters_path}")
