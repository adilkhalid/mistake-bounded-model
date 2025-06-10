class ConjunctionLearner:
    def __init__(self, n_features):
        self.hypothesis = set()
        self.n_features = n_features

        # start with all possible literal
        for i in range(n_features):
            self.hypothesis.add((i, True))
            self.hypothesis.add((i, False))

    def predict(self, x):
        for index, is_positive in self.hypothesis:
            if is_positive and x[index] == 0:
                return 0
            if not is_positive and x[index] == 1:
                return 0
        return 1

    def update(self, x, y_true):
        y_pred = self.predict(x)
        if y_pred == y_true:
            if y_true == 1:
                to_remove = set()
                for index, is_positive in self.hypothesis:
                    if (is_positive and x[index] == 0) or (not is_positive and x[index] == 1):
                        to_remove.add((index, is_positive))
                self.hypothesis -= to_remove
        return y_pred