from sklearn.ensemble import RandomForestClassifier

class Model:
    def __init__(self):
        self.clf = RandomForestClassifier(max_depth=2, random_state=0)

    def fit(self, X, y):
        self.clf.fit(X=X, y=y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_score(self, X):
        return self.clf.predict_proba(X)