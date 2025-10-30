import numpy as np

class OnlineFuzzyART:
    def __init__(self, input_dim, vigilance=0.5, choice_smoothing=0.01, learning_rate=1.0):
        self.input_dim = input_dim
        self.rho = vigilance
        self.alpha = choice_smoothing
        self.beta = learning_rate
        self.weights = np.empty((0, self.input_dim), dtype=np.float32)

    def _normalize(self, x):
        max_val = np.max(x)
        return x / (max_val + 1e-6)

    def learn(self, x):
        """
        Learn input x and return:
        - assigned category index
        - novelty score = 1 - match_score
        """
        x_norm = self._normalize(x)
        
        if self.weights.shape[0] == 0:
            # Add the first weight vector
            self.weights = np.vstack([self.weights, x_norm])
            return 0, 1.0

        fuzzy_and = np.minimum(self.weights, x_norm)
        sum_fuzzy_and = np.sum(fuzzy_and, axis=1)
        sum_weights = np.sum(self.weights, axis=1)
        scores = sum_fuzzy_and / (self.alpha + sum_weights)
        
        sorted_indices = np.argsort(scores)[::-1]
        best_score = scores[sorted_indices[0]]

        sum_x = np.sum(x_norm)
        vigilance_pass = (sum_fuzzy_and / (sum_x + 1e-8)) >= self.rho

        for j in sorted_indices:
            if vigilance_pass[j]:
                self.weights[j] = self.beta * fuzzy_and[j] + (1 - self.beta) * self.weights[j]
                return j, 1.0 - scores[j]

        self.weights = np.vstack([self.weights, x_norm])
        return self.weights.shape[0] - 1, 1.0 - best_score

    def predict(self, x):
        """
        Predict category and return:
        - index of best-matching category (or -1 if none passes vigilance)
        - novelty score = 1 - match_score
        """
        x_norm = self._normalize(x)

        if self.weights.shape[0] == 0:
            return -1, 1.0
        
        # --- Vectorized Calculation (same as in learn) ---
        fuzzy_and = np.minimum(self.weights, x_norm)
        sum_fuzzy_and = np.sum(fuzzy_and, axis=1)
        sum_weights = np.sum(self.weights, axis=1)
        scores = sum_fuzzy_and / (self.alpha + sum_weights)
        
        sorted_indices = np.argsort(scores)[::-1]
        best_score = scores[sorted_indices[0]]
        
        sum_x = np.sum(x_norm)
        vigilance_pass = (sum_fuzzy_and / (sum_x + 1e-8)) >= self.rho
        
        for j in sorted_indices:
            if vigilance_pass[j]:
                return j, 1.0 - scores[j]
                
        return -1, 1.0 - best_score

    def num_categories(self):
        return self.weights.shape[0]