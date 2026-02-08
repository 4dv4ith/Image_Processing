#Download any OCR dataset and perform the classification with SVM and KNN. Compare the obtained result.

# OCR Classification using SVM and KNN
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Step 1: Load OCR dataset
digits = load_digits()
X = digits.data        # image features (64 values per image)
y = digits.target      # corresponding labels (0–9)

# Step 2: Train–test splitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3: Train SVM Classifier
svm_model = SVC(kernel='rbf', gamma='scale')
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
svm_acc = accuracy_score(y_test, svm_pred)

# Step 4: Train KNN Classifier
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)
knn_acc = accuracy_score(y_test, knn_pred)

print("📌 Accuracy Comparison")
print("----------------------")
print(f"SVM Accuracy : {svm_acc:.4f}")
print(f"KNN Accuracy : {knn_acc:.4f}")

print("\n🔍 SVM Classification Report:")
print(classification_report(y_test, svm_pred))

print("\n🔍 KNN Classification Report:")
print(classification_report(y_test, knn_pred))
