import joblib
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.calibration import CalibratedClassifierCV

from src.utils.logger import get_logger
from src.config import MODELS_DIR

logger = get_logger(__name__)


# =====================================================
# FEATURE IMPORTANCE
# =====================================================
def get_top_features(model, vectorizer, label_encoder, top_n=20):

    if not hasattr(model, "calibrated_classifiers_"):
        logger.warning("Model is not calibrated. Cannot extract features.")
        return

    calibrated_model = model.calibrated_classifiers_[0]

    if not hasattr(calibrated_model, "estimator"):
        logger.warning("Could not access underlying estimator.")
        return

    lr_model = calibrated_model.estimator

    if not hasattr(lr_model, "coef_"):
        logger.warning("Underlying model has no coefficients.")
        return

    feature_names = np.array(vectorizer.get_feature_names_out())
    classes = label_encoder.classes_

    logger.info("\n========== TOP FEATURES PER CLASS ==========")

    for i, class_label in enumerate(classes):
        top_indices = np.argsort(lr_model.coef_[i])[-top_n:]
        top_features = feature_names[top_indices]

        logger.info(f"\nTop features for class '{class_label}':")
        for feature in reversed(top_features):
            logger.info(feature)


# =====================================================
# TRAIN BASELINE
# =====================================================
def train_baseline(df):

    logger.info("Starting baseline training...")

    X = df["Document"]
    y = df["Topic_group"]

    # -------------------------
    # Encode labels
    # -------------------------
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # -------------------------
    # Train / Val / Test Split
    # -------------------------
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y_encoded,
        test_size=0.30,
        stratify=y_encoded,
        random_state=42,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=42,
    )

    logger.info(f"Train size: {len(X_train)}")
    logger.info(f"Validation size: {len(X_val)}")
    logger.info(f"Test size: {len(X_test)}")

    # -------------------------
    # TF-IDF Vectorization
    # -------------------------
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=50000,
        min_df=2,
        max_df=0.9,
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)

    # -------------------------
    # Logistic Regression
    # -------------------------
    base_model = LogisticRegression(
        solver="saga",
        max_iter=5000,
        class_weight="balanced",
        C=0.7,
        n_jobs=-1,
    )

    # -------------------------
    # Calibration
    # -------------------------
    model = CalibratedClassifierCV(
        base_model,
        method="sigmoid",
        cv=3,
    )

    model.fit(X_train_vec, y_train)

    # =====================================================
    # VALIDATION EVALUATION
    # =====================================================
    y_val_pred = model.predict(X_val_vec)

    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average="macro")

    logger.info(f"\nValidation Accuracy: {val_acc:.4f}")
    logger.info(f"Validation Macro F1: {val_f1:.4f}")
    logger.info(
        "\nValidation Classification Report:\n"
        + classification_report(y_val, y_val_pred)
    )

    # Validation confidence
    val_probs = model.predict_proba(X_val_vec)
    val_confidence = np.max(val_probs, axis=1)

    logger.info(f"\nAverage validation confidence: {np.mean(val_confidence):.4f}")
    logger.info(
        f"% predictions > 0.80 confidence: "
        f"{(val_confidence > 0.80).mean() * 100:.2f}%"
    )

    high_conf = val_confidence > 0.85
    medium_conf = (val_confidence > 0.6) & (val_confidence <= 0.85)
    low_conf = val_confidence <= 0.6

    logger.info(f"Auto-route (high confidence): {high_conf.mean() * 100:.2f}%")
    logger.info(f"Manual review (medium confidence): {medium_conf.mean() * 100:.2f}%")
    logger.info(f"Fallback queue (low confidence): {low_conf.mean() * 100:.2f}%")

    # =====================================================
    # TEST EVALUATION
    # =====================================================
    y_test_pred = model.predict(X_test_vec)

    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average="macro")

    logger.info(f"\nTest Accuracy: {test_acc:.4f}")
    logger.info(f"Test Macro F1: {test_f1:.4f}")
    logger.info(
        "\nTest Classification Report:\n" + classification_report(y_test, y_test_pred)
    )

    # -------------------------
    # TEST CONFIDENCE ROUTING
    # -------------------------
    test_probs = model.predict_proba(X_test_vec)
    test_confidence = np.max(test_probs, axis=1)

    logger.info(f"\nAverage test confidence: {np.mean(test_confidence):.4f}")
    logger.info(
        f"Test auto-route (>0.85): " f"{(test_confidence > 0.85).mean() * 100:.2f}%"
    )
    logger.info(
        f"Test fallback (<0.6): " f"{(test_confidence <= 0.6).mean() * 100:.2f}%"
    )

    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================
    get_top_features(model, vectorizer, label_encoder)

    # =====================================================
    # SAVE ARTIFACTS
    # =====================================================
    baseline_dir = MODELS_DIR / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, baseline_dir / "model.pkl")
    joblib.dump(vectorizer, baseline_dir / "vectorizer.pkl")
    joblib.dump(label_encoder, baseline_dir / "label_encoder.pkl")

    logger.info("Calibrated baseline model saved successfully.")

    return model, vectorizer, label_encoder
