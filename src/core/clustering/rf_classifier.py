"""Stage B 판정기: best(1) / not-best(0) 2클래스 Random Forest.

cuML이 있으면 GPU, 없으면 scikit-learn(CPU)을 사용한다.
"""

import time

import numpy as np

try:
    import cupy as cp
    from cuml.ensemble import RandomForestClassifier as _CuRF
    USE_CUML = True
except ImportError:
    cp = None
    _CuRF = None
    USE_CUML = False
from sklearn.ensemble import RandomForestClassifier as _SkRF

NOT_BEST, BEST = 0, 1


class StageBForest:
    """best / not-best Random Forest. 학습셋은 volume_classification.train_stage_b_forest가 만든다."""

    def __init__(self, n_estimators=100, min_samples_split=8, random_state=42):
        self.n_estimators = int(n_estimators)
        self.min_samples_split = int(min_samples_split)
        self.random_state = random_state
        self.use_gpu = USE_CUML
        self.inr_only = False   # True면 INR embedding + coords 입력으로 학습/판정        self.model = None
        self.classes_ = None
        self.train_info = {}

    def _make_model(self):
        if self.use_gpu:
            return _CuRF(n_estimators=self.n_estimators, random_state=self.random_state,
                         min_samples_split=self.min_samples_split, n_streams=4,
                         max_features='sqrt', bootstrap=False)
        return _SkRF(n_estimators=self.n_estimators, random_state=self.random_state,
                     min_samples_split=self.min_samples_split, max_features='sqrt',
                     bootstrap=False, n_jobs=-1)

    def fit(self, X, y, extra_info=None):
        """X (N, D) feature, y (N,) NOT_BEST/BEST로 학습한다.

        train_info에 학습 샘플 수, 학습셋 정확도, backend, extra_info를 기록한다.
        Returns: self
        """
        t0 = time.time()
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(np.int32)
        if len(np.unique(y)) < 2:
            raise ValueError("RF 학습셋에 best/not-best 중 한쪽 라벨이 없음")

        self.model = self._make_model()
        if self.use_gpu:
            self.model.fit(cp.asarray(X), cp.asarray(y))
            self.classes_ = np.unique(y)
        else:
            self.model.fit(X, y)
            self.classes_ = np.asarray(self.model.classes_, dtype=np.int32)

        train_acc = float(np.mean(self.predict_classes(X) == y))
        self.train_info = {"n_train": int(len(y)), "n_best": int(np.sum(y == BEST)),
                           "n_not_best": int(np.sum(y == NOT_BEST)),
                           "train_accuracy": train_acc, "n_estimators": self.n_estimators,
                           "backend": "cuML" if self.use_gpu else "sklearn",
                           **(extra_info or {}),
                           "elapsed_sec": time.time() - t0}
        print(f"🌲 [Stage B RF] 학습 best {self.train_info['n_best']:,} / not-best "
              f"{self.train_info['n_not_best']:,}, tree {self.n_estimators}, "
              f"{self.train_info['backend']}, 학습셋 정확도 {train_acc:.3f}, "
              f"{self.train_info['elapsed_sec']:.1f}초")
        return self

    def predict_proba(self, X):
        """Returns: (M, len(classes_)) float32 클래스 확률 (열 순서 = classes_)"""
        X = np.asarray(X, dtype=np.float32)
        if len(X) == 0:
            return np.zeros((0, len(self.classes_)), dtype=np.float32)
        if self.use_gpu:
            return cp.asnumpy(self.model.predict_proba(cp.asarray(X))).astype(np.float32)
        return self.model.predict_proba(X).astype(np.float32)

    def predict_classes(self, X):
        """Returns: (M,) 예측 클래스"""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def predict_best_proba(self, X):
        """Returns: (M,) P(best)"""
        return self.predict_proba(X)[:, int(np.flatnonzero(self.classes_ == BEST)[0])]


def rf_decision(p_best, min_best_prob=0.0):
    """best 판정. Returns: (M,) bool — P(best) > 0.5 이고 P(best) >= min_best_prob"""
    p_best = np.asarray(p_best)
    return (p_best > 0.5) & (p_best >= min_best_prob)
