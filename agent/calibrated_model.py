# =============================================================================
# MAZU — wrapper combining a base classifier with a fitted isotonic
# calibrator, so the combination can be joblib-dumped/loaded and called with
# the same predict_proba(X) interface forecast_tool already expects.
#
# Defined in its own module (not inline in a script) because joblib/pickle
# requires the class to be importable from the same module path at load
# time as it was at save time.
# =============================================================================
import numpy as np


class CalibratedModel:
    def __init__(self, base_clf, calibrator):
        self.base_clf = base_clf
        self.calibrator = calibrator

    def predict_proba(self, X):
        raw = self.base_clf.predict_proba(X)[:, 1]
        calibrated = np.clip(self.calibrator.predict(raw), 0.0, 1.0)
        return np.column_stack([1.0 - calibrated, calibrated])
