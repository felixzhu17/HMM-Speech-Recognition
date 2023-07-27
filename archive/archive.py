import scipy.stats as st
import numpy as np


class BatchStableGMMHMM:
    def __init__(self, n_states, n_dims, deterministic_start = True):
        self.n_states = n_states
        self.n_dims = n_dims
        self.deterministic_start = deterministic_start
        self.random_state = np.random.RandomState(0)
        
        # Normalize random initial state

        if self.deterministic_start:
            self.prior = np.zeros((self.n_states, 1)) + 1e-6
            self.prior[0] = 1
            self.prior = self._normalize(self.prior)

        else:
            self.prior = self._normalize(self.random_state.rand(self.n_states, 1))
            
        self.A = self._stochasticize(self.random_state.rand(self.n_states, self.n_states))
        self.log_prior = np.log(self.prior)
        self.log_A = np.log(self.A)

        self.mu = None
        self.covs = None

    def _init_mu(self, obs):
        # Randomly select n_states observations from obs as initial mean estimates
        flattened_obs = obs.reshape(obs.shape[1], -1)
        subset = self.random_state.choice(np.arange(flattened_obs.shape[1]), size=self.n_states, replace=False)
        self.mu = flattened_obs[:, subset]

    def _init_covs(self, obs):
        # Use sample covariances of observations as initial covariance estimates, however, make them diagonal
        flattened_obs = obs.reshape(obs.shape[1], -1)
        self.covs = np.zeros((self.n_states, self.n_dims, self.n_dims))
        for i in range(self.n_states):
            self.covs[i] += np.diag(np.diag(np.cov(flattened_obs)))
            
    def log_likelihood(self, obs):
        self._check_obs(obs)
        log_likelihood, _ = self.forward(obs)
        return log_likelihood

    def em_step(self, obs): 
        self._check_obs(obs)
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
        self.gamma = self.alpha + self.beta - log_likelihood.reshape(-1, 1, 1)

        # Compute the probability of being in state i at time t and state j at time t+1
        n_batches, n_observations = self._get_batches_observations(obs)
        self.xi = np.zeros((n_batches, self.n_states, self.n_states, n_observations - 1))
        for batch in range(n_batches):
            for t in range(n_observations - 1): # Loop through every time step
                for i in range(self.n_states): # Loop through every start state
                    # Compute transition probability 
                    self.xi[batch, i, :, t] = self.alpha[batch, i, t]  + self.log_A[i, :] + emission_log_likelihood[batch, :, t + 1] + self.beta[batch, :, t + 1]
        return log_likelihood, self.gamma

    def m_step(self, obs):
        # Prior is the gamma along the time axis at 0
        n_batches, n_observations = self._get_batches_observations(obs)
        self.log_prior = np.logaddexp.reduce((self.gamma[:, :, 0] - np.log(n_batches))) # Take average of exponential of log probabilities
        self.prior = np.exp(self.log_prior)

        # Stable representation of gamma. Weighted averages are invariant to scaling of probabilities
        gamma_s = np.exp(self.gamma - np.max(self.gamma))

        # Mean is the weighted sum of each observation by the probability of being in that state
        for s in range(self.n_states):
            self.mu[:,s] = ((gamma_s[:, s, :].reshape(n_batches, 1, n_observations) * obs).sum(axis=2).sum(axis=0))/gamma_s[:, s, :].sum()

        # Covariance is the weighted sum of the outer product of the difference of each observation from the mean
        for s in range(self.n_states):
            diff = obs - self.mu[:, s].reshape(-1, 1)
            self.covs[s, :, :] = np.matmul(gamma_s[:, s, :].reshape(n_batches, 1, n_observations) * diff, diff.transpose(0, 2, 1)).sum(axis=0) / gamma_s[:, s, :].sum()
            self.covs[s, :, :] += np.eye(self.n_dims) * 1e-6  # Add a small value to avoid numerical underflow

        # Transition matrix is the expected number of transitions from state i to state j
        for i in range(self.n_states):
            for j in range(self.n_states):
                self.A[i, j] = np.exp(np.logaddexp.reduce(self.xi[:, i, j, :].flatten()) - np.logaddexp.reduce(self.gamma[:, i, :-1].flatten()))
        self.A = self._stochasticize(self.A)
        self.log_A = np.log(self.A)
        return

    def _get_emission_log_likelihood(self, obs):
        n_batches, n_observations = self._get_batches_observations(obs)
        B = np.zeros((n_batches, self.n_states, n_observations))
        for batch in range(n_batches):
            for s in range(self.n_states):
                np.random.seed(self.random_state.randint(1))
                B[batch, s, :] = st.multivariate_normal.pdf(
                    obs[batch].T, mean=self.mu[:, s].T, cov=self.covs[s, :, :].T)
        return np.log(B)

    def forward(self, obs, emission_log_likelihood = None):
        if emission_log_likelihood is None:
            emission_log_likelihood = self._get_emission_log_likelihood(obs)
        alpha = np.zeros(emission_log_likelihood.shape)
        n_observations = alpha.shape[2]

        # First time step is equal to prior state times likelihood
        alpha[:, :, 0] = emission_log_likelihood[:, :, 0] + self.log_prior.ravel()

        for t in range(1, n_observations):
            # Subsequent time steps are equal to sum of transitions into state times likelihood
            # Add log probabilitites of transitions and historical alpha before converting to normal space to avoid numerical underflow
            alpha[:, :, t] = emission_log_likelihood[:, :, t] + np.logaddexp.reduce(self.log_A.T + alpha[:, np.newaxis, :, t - 1], axis=2)  # Insert additional axis to broadcast along the rows of the transition matrix

        log_likelihood = np.logaddexp.reduce(alpha[:, :, -1], axis = 1)
        
        return log_likelihood, alpha
    
    def backward(self, obs, emission_log_likelihood = None):
        if emission_log_likelihood is None:
            emission_log_likelihood = self._get_emission_log_likelihood(obs)
        beta = np.zeros(emission_log_likelihood.shape)
        n_observations = beta.shape[2]

        # Last time step is equal to one as there are no future observations
        beta[:, :, -1] = np.zeros(self.n_states)

        for t in range(n_observations - 1)[::-1]:
            # # Previous time steps are equal to sum of transitions from state times likelihood
            # # Add log probabilitites of transitions and future beta before converting to normal space to avoid numerical underflow
            beta[:, :, t] = np.logaddexp.reduce(beta[:, np.newaxis, :, t+1] + emission_log_likelihood[:, np.newaxis, :, t+1] + self.log_A, axis=2) # Insert additional axis to broadcast along the rows of the transition matrix
            
        return beta
    
    def _normalize(self, x):
        # Ensure sum to 1
        return (x + (x == 0)) / np.sum(x)
    
    def _stochasticize(self, x):
        # Ensure rows sum to 1
        return x / np.sum(x, axis=1)[:, np.newaxis]

    def _get_batches_observations(self, obs):
        return obs.shape[0], obs.shape[2]

    def _check_obs(self, obs):
        assert len(obs.shape) == 3, "obs must have three dimensions"
        assert obs.shape[1] == self.n_dims, f"Second dimension of obs must be equal to {self.n_dims}"
