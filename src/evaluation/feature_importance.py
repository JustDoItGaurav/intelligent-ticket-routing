import numpy as np

def get_top_features(model, vectorizer, label_encoder, top_n=10):
    feature_names = vectorizer.get_feature_names_out()

    # Get base estimator inside calibrated model
    base_model = model.base_estimator

    for i, class_label in enumerate(label_encoder.classes_):
        top_features = np.argsort(base_model.coef_[i])[-top_n:]
        print(f"\nTop features for class '{class_label}':")
        for idx in reversed(top_features):
            print(feature_names[idx])