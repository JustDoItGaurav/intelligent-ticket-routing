import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_confusion(model, X, y, vectorizer, label_encoder):
    X_vec = vectorizer.transform(X)
    y_pred = model.predict(X_vec)

    cm = confusion_matrix(y, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=label_encoder.classes_
    )
    disp.plot(xticks_rotation=45)
    plt.show()
