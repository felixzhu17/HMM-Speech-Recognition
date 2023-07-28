import scipy.stats as st
import numpy as np


class StableGMMHMM:
    def __init__(self, n_states, n_dims):
        self.n_states = n_states
        self.n_dims = n_dims
        self.random_state = np.random.RandomState(0)

        # Normalize random initial state
        self.prior = self._normalize(self.random_state.rand(self.n_states, 1))
        self.A = self._stochasticize(
            self.random_state.rand(self.n_states, self.n_states)
        )
        self.log_prior = np.log(self.prior)
        self.log_A = np.log(self.A)

        self.mu = None
        self.covs = None

    def _init_mu(self, obs):
        # Randomly select n_states observations from obs as initial mean estimates
        subset = self.random_state.choice(
            np.arange(obs.shape[1]), size=self.n_states, replace=False
        )
        self.mu = obs[:, subset]

    def _init_covs(self, obs):
        # Use sample covariances of observations as initial covariance estimates, however, make them diagonal
        self.covs = np.zeros((self.n_states, self.n_dims, self.n_dims))
        for i in range(self.n_states):
            self.covs[i] += np.diag(np.diag(np.cov(obs)))

    def log_likelihood(self, obs):
        log_likelihood, _ = self.forward(obs)
        return log_likelihood

    def em_step(self, obs):
        if self.mu is None:
            print("Initializing mu...")
            self._init_mu(obs)
        if self.covs is None:
            print("Initializing covs...")
            self._init_covs(obs)
        log_likelihood, _ = self.e_step(obs)
        self.m_step(obs)
        return log_likelihood

    def e_step(self, obs):
        # Get the likelihood of obs under each state
        emission_log_likelihood = self._get_emission_log_likelihood(obs)

        # Compute Forward Variables
        log_likelihood, self.alpha = self.forward(obs, emission_log_likelihood)

        # Compute Backward Variables
        self.beta = self.backward(obs, emission_log_likelihood)

        # Compute Gammas
        self.gamma = self.alpha + self.beta - log_likelihood

        # Compute the probability of being in state i at time t and state j at time t+1
        n_observations = obs.shape[1]
        self.xi = np.zeros((self.n_states, self.n_states, n_observations - 1))
        for t in range(n_observations - 1):  # Loop through every time step
            for i in range(self.n_states):  # Loop through every start state
                # Compute transition probability
                self.xi[i, :, t] = (
                    self.alpha[i, t]
                    + self.log_A[i, :]
                    + emission_log_likelihood[:, t + 1]
                    + self.beta[:, t + 1]
                )

        return log_likelihood, self.gamma

    def m_step(self, obs):
        # Prior is the gamma along the time axis at 0
        self.log_prior = self.gamma[:, 0]
        self.prior = np.exp(self.log_prior)

        # Stable representation of gamma. Weighted averages are invariant to scaling of probabilities
        gamma_s = np.exp(self.gamma - np.max(self.gamma))

        # Mean is the weighted sum of each observation by the probability of being in that state
        for s in range(self.n_states):
            self.mu[:, s] = ((gamma_s[s] * obs).sum(axis=1)) / gamma_s[s].sum()

        # Covariance is the weighted sum of the outer product of the difference of each observation from the mean
        for s in range(self.n_states):
            diff = obs - self.mu[:, s].reshape(-1, 1)
            self.covs[s, :, :] = np.dot(gamma_s[s] * diff, diff.T) / gamma_s[s].sum()
            self.covs[s, :, :] += (
                np.eye(self.n_dims) * 1e-6
            )  # Add a small value to avoid numerical underflow

        # Transition matrix is the expected number of transitions from state i to state j
        for i in range(self.n_states):
            for j in range(self.n_states):
                self.A[i, j] = np.exp(
                    np.logaddexp.reduce(self.xi[i, j, :])
                    - np.logaddexp.reduce(self.gamma[i, :-1])
                )
        self.A = self._stochasticize(self.A)
        self.log_A = np.log(self.A)
        return

    def _get_emission_log_likelihood(self, obs):
        B = np.zeros((self.n_states, obs.shape[1]))
        for s in range(self.n_states):
            np.random.seed(self.random_state.randint(1))
            B[s, :] = st.multivariate_normal.pdf(
                obs.T, mean=self.mu[:, s].T, cov=self.covs[s, :, :].T
            )
        return np.log(B)

    def forward(self, obs, emission_log_likelihood=None):
        if emission_log_likelihood is None:
            emission_log_likelihood = self._get_emission_log_likelihood(obs)
        alpha = np.zeros(emission_log_likelihood.shape)
        n_observations = alpha.shape[1]
        for t in range(n_observations):
            if t == 0:
                # First time step is equal to prior state times likelihood
                alpha[:, t] = emission_log_likelihood[:, t] + self.log_prior.ravel()
            else:
                # Subsequent time steps are equal to sum of transitions into state times likelihood
                # Add log probabilitites of transitions and historical alpha before converting to normal space to avoid numerical underflow
                alpha[:, t] = emission_log_likelihood[:, t] + np.logaddexp.reduce(
                    self.log_A.T + alpha[:, t - 1], axis=1
                )

        log_likelihood = np.logaddexp.reduce(alpha[:, -1])
        return log_likelihood, alpha

    def backward(self, obs, emission_log_likelihood=None):
        if emission_log_likelihood is None:
            emission_log_likelihood = self._get_emission_log_likelihood(obs)
        beta = np.zeros(emission_log_likelihood.shape)
        n_observations = beta.shape[1]
        # Last time step is equal to one as there are no future observations
        beta[:, -1] = np.zeros(self.n_states)
        for t in range(n_observations - 1)[::-1]:
            # # Previous time steps are equal to sum of transitions from state times likelihood
            # # Add log probabilitites of transitions and future beta before converting to normal space to avoid numerical underflow
            beta[:, t] = np.logaddexp.reduce(
                (beta[:, t + 1] + emission_log_likelihood[:, t + 1]) + self.log_A,
                axis=1,
            )
        return beta

    def _normalize(self, x):
        # Ensure sum to 1
        return (x + (x == 0)) / np.sum(x)

    def _stochasticize(self, x):
        # Ensure rows sum to 1
        return x / np.sum(x, axis=1)[:, np.newaxis]


class RawGMMHMM:
    def __init__(self, n_states, n_dims):
        self.n_states = n_states
        self.n_dims = n_dims
        self.random_state = np.random.RandomState(0)

        # Normalize random initial state
        self.prior = self._normalize(self.random_state.rand(self.n_states, 1))
        self.A = self._stochasticize(
            self.random_state.rand(self.n_states, self.n_states)
        )

        self.mu = None
        self.covs = None

    def _init_mu(self, obs):
        # Randomly select n_states observations from obs as initial mean estimates
        subset = self.random_state.choice(
            np.arange(obs.shape[1]), size=self.n_states, replace=False
        )
        self.mu = obs[:, subset]

    def _init_covs(self, obs):
        # Use sample covariances of observations as initial covariance estimates, however, make them diagonal
        self.covs = np.zeros((self.n_states, self.n_dims, self.n_dims))
        for i in range(self.n_states):
            self.covs[i] += np.diag(np.diag(np.cov(obs)))

    def likelihood(self, obs):
        likelihood, _ = self.forward(obs)
        return likelihood

    def em_step(self, obs):
        if self.mu is None:
            print("Initializing mu...")
            self._init_mu(obs)
        if self.covs is None:
            print("Initializing covs...")
            self._init_covs(obs)
        likelihood, _ = self.e_step(obs)
        self.m_step(obs)
        return likelihood

    def e_step(self, obs):
        # Get the likelihood of obs under each state
        emission_likelihood = self._get_emission_likelihood(obs)

        # Compute Forward Variables
        likelihood, self.alpha = self.forward(obs, emission_likelihood)

        # Compute Backward Variables
        self.beta = self.backward(obs, emission_likelihood)

        # Compute Gammas
        self.gamma = (self.alpha * self.beta) / likelihood

        # Compute the probability of being in state i at time t and state j at time t+1
        n_observations = obs.shape[1]
        self.xi = np.zeros((self.n_states, self.n_states, n_observations - 1))
        for t in range(n_observations - 1):  # Loop through every time step
            for i in range(self.n_states):  # Loop through every start state
                # Compute transition probability
                self.xi[i, :, t] = (
                    self.alpha[i, t]
                    * self.A[i, :]
                    * emission_likelihood[:, t + 1]
                    * self.beta[:, t + 1]
                    / likelihood
                )
        return likelihood, self.gamma

    def m_step(self, obs):
        # Prior is the gamma along the time axis at 0
        self.prior = self.gamma[:, 0]

        # Mean is the weighted sum of each observation by the probability of being in that state
        for s in range(self.n_states):
            self.mu[:, s] = ((self.gamma[s] * obs).sum(axis=1)) / self.gamma[s].sum()

        # Covariance is the weighted sum of the outer product of the difference of each observation from the mean
        for s in range(self.n_states):
            diff = obs - self.mu[:, s].reshape(-1, 1)
            self.covs[s, :, :] = (
                np.dot(self.gamma[s] * diff, diff.T) / self.gamma[s].sum()
            )
            self.covs[s, :, :] += (
                np.eye(self.n_dims) * 1e-6
            )  # Add a small value to avoid numerical underflow

        # Transition matrix is the expected number of transitions from state i to state j
        for i in range(self.n_states):
            for j in range(self.n_states):
                self.A[i, j] = np.sum(self.xi[i, j, :]) / self.gamma[i, :-1].sum()
        return

    def _get_emission_likelihood(self, obs):
        B = np.zeros((self.n_states, obs.shape[1]))
        for s in range(self.n_states):
            np.random.seed(self.random_state.randint(1))
            B[s, :] = st.multivariate_normal.pdf(
                obs.T, mean=self.mu[:, s].T, cov=self.covs[s, :, :].T
            )
        return B

    def forward(self, obs, emission_likelihood=None):
        if emission_likelihood is None:
            emission_likelihood = self._get_emission_likelihood(obs)
        alpha = np.zeros(emission_likelihood.shape)
        n_observations = alpha.shape[1]
        for t in range(n_observations):
            if t == 0:
                # First time step is equal to prior state times likelihood
                alpha[:, t] = emission_likelihood[:, t] * self.prior.ravel()
            else:
                # Subsequent time steps are equal to sum of transitions into state times likelihood
                alpha[:, t] = emission_likelihood[:, t] * self.A.T @ alpha[:, t - 1]

        # Sum probabilities of last time step is the marginal distirbution of all observations
        likelihood = alpha[:, -1].sum()
        return likelihood, alpha

    def backward(self, obs, emission_likelihood=None):
        if emission_likelihood is None:
            emission_likelihood = self._get_emission_likelihood(obs)
        beta = np.zeros(emission_likelihood.shape)
        n_observations = beta.shape[1]
        # Last time step is equal to one as there are no future observations
        beta[:, -1] = np.ones(self.n_states)
        for t in range(n_observations - 1)[::-1]:
            # Previous time steps are equal to sum of transitions from state times likelihood
            beta[:, t] = self.A @ (emission_likelihood[:, t + 1] * beta[:, t + 1])
        return beta

    def _normalize(self, x):
        # Ensure sum to 1
        return (x + (x == 0)) / np.sum(x)

    def _stochasticize(self, x):
        # Ensure rows sum to 1
        return x / np.sum(x, axis=1)[:, np.newaxis]
