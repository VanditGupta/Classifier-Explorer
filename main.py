from sklearn.model_selection import train_test_split
import streamlit as st
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Set the Title of the Webpage
# title = st.title("Streamlit Tutorial")

# Header
st.write("""
         # Exploring different classifiers
         ## Which one is the best ?
         """)

# Select box on page
# dataset_name = st.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine Dataset", "MNIST"))

# Select Dataset on sidebar
dataset_name = st.sidebar.selectbox(
    "Select Dataset", ("Iris", "Breast Cancer", "Wine Dataset", "MNIST"))

# Select Classifier on sidebar
classifier_name = st.sidebar.selectbox(
    "Select Classifier", ("KNN", "SVM", "Random Forest"))

# st.write(dataset_name)


def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    elif dataset_name == "Wine Dataset":
        data = datasets.load_wine()
    else:
        data = datasets.load_digits()

    X = data.data
    y = data.target
    return X, y


X, y = get_dataset(dataset_name)

st.write("Shape of Dataset", X.shape)
st.write("Number of Classes", len(set(y)))


# Add Parameters to sidebar

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return params


params = add_parameter_ui(classifier_name)

# Add Classifier
def get_classifier(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "SVM":
        clf = SVC(C=params["C"])
    else:
        clf = RandomForestClassifier(
            n_estimators=params["n_estimators"], max_depth=params["max_depth"], random_state=1234)
    return clf


clf = get_classifier(classifier_name, params)

st.write(f"Classifier = {classifier_name}")

# Classification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234)

clf.fit(X_train, y_train)

# Print Training Accuracy

y_pred = clf.predict(X_train)
acc = accuracy_score(y_train, y_pred)
st.write(f"Training Accuracy = {acc}")

# Print Test Accuracy

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.write(f"Test Accuracy = {acc}")

# Plot the results based on the test accuracy

# PLOT

# Plot the results in a 2D plane
pca = PCA(2)
X_projected = pca.fit_transform(X)

# Values
x1 = X_projected[:, 0]

# Labels
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()

# plt.show()

st.pyplot(fig)


