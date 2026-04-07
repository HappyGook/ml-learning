import numpy as np


class Model:
    def __init__(self,k,X):
        self.k = k
        self.X = X
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.Rs = np.array([[]])
        self.means = X[np.random.choice(self.n,self.k,replace=False)]
        self.covariances = np.array([np.eye(self.d) for _ in range(self.k)])
        self.weights = np.ones(self.k)/ self.k

    # method to compute gaussians efficiently (ish)
    def compute_log_gaussian(self):

        log_probs = np.zeros((self.n, self.k))

        for k in range(self.k):
            diff = self.X - self.means[k]  # (N, D)
            inv = np.linalg.inv(self.covariances[k])  # (D, D)
            log_det = np.linalg.slogdet(self.covariances[k])[1]

            quad = np.sum(diff @ inv * diff, axis=1)  # (N,)

            log_probs[:, k] = -0.5 * (quad + log_det + self.d * np.log(2 * np.pi))

        return log_probs

    def e_step(self):
        # everything is computed in log-space
        log_probs = self.compute_log_gaussian()

        # weights[k] * gaussian_density
        log_probs += np.log(self.weights) # add the priors (weights)

        # take the largest component score for each data point
        max_log = np.max(log_probs, axis=1, keepdims=True)

        # denominator
        log_sum = max_log + np.log(np.sum(np.exp(log_probs - max_log), axis=1, keepdims=True))

        log_resp = log_probs - log_sum

        # exit log space via exponentiating
        self.Rs = np.exp(log_resp)


    def m_step(self):

        # update weights
        nk = np.sum(self.Rs, axis=0) # (K,)
        self.weights = nk / self.n

        #update means
        self.means = (self.Rs.T @ self.X) / nk[:,None] # for (K,D)

        #update covariances
        for k in range(self.k):
            diff = self.X - self.means[k]
            self.covariances[k]=(self.Rs[:,k][:,None]*diff).T @ diff / nk[k]
            self.covariances[k] +=1e-6 * np.eye(self.d)
