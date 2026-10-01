"""Manual SMOTE-style oversampling pipeline (imblearn unavailable, no network to install).
k-NN interpolation between minority-class neighbors in preprocessed, train-fold-only space.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors


class SmotePipeline:
    def __init__(self, preprocess, clf):
        self.preprocess = preprocess
        self.clf = clf
        self.named_steps = {"prep": preprocess, "clf": clf}

    def fit(self, X, y, k=5, random_state=42):
        rng = np.random.RandomState(random_state)
        Xt = self.preprocess.fit_transform(X)
        Xt = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
        y = np.asarray(y)
        maj_count = (y == 0).sum()
        min_count = (y == 1).sum()
        n_to_generate = maj_count - min_count
        min_X = Xt[y == 1]
        nn = NearestNeighbors(n_neighbors=min(k + 1, len(min_X))).fit(min_X)
        _, neighbor_idx = nn.kneighbors(min_X)
        synthetic = []
        for _ in range(n_to_generate):
            i = rng.randint(0, len(min_X))
            neighbors = neighbor_idx[i][1:]
            j = neighbors[rng.randint(0, len(neighbors))]
            gap = rng.rand()
            synthetic.append(min_X[i] + gap * (min_X[j] - min_X[i]))
        synthetic = np.array(synthetic)
        X_res = np.vstack([Xt, synthetic])
        y_res = np.concatenate([y, np.ones(len(synthetic), dtype=int)])
        self.clf.fit(X_res, y_res)
        return self

    def predict(self, X):
        Xt = self.preprocess.transform(X)
        Xt = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
        return self.clf.predict(Xt)

    def predict_proba(self, X):
        Xt = self.preprocess.transform(X)
        Xt = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
        return self.clf.predict_proba(Xt)
