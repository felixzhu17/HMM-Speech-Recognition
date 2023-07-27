import scipy.stats as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DNN(nn.Module):
    def __init__(self, input_dim, output_dim, encoding_dim):
        super().__init__()
        #self.lstm = nn.LSTM(input_size=input_dim, hidden_size=encoding_dim, batch_first=True, bidirectional =True)
        self.fc1 = nn.Linear(input_dim, encoding_dim) 
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(encoding_dim, output_dim)  

    def forward(self, x):
        #x = F.relu(self.lstm(x)[0])
        x = F.relu(self.fc1(x))  # Apply ReLU activation function after the first layer
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)  # Apply the second layer
        return F.log_softmax(x, dim=-1)  # Apply log softmax to make output probabilities

class DNNHMM:
    def __init__(self, n_states, n_dims, n_encoding_dims = None, deterministic_start = True):
        self.n_states = n_states
        self.n_dims = n_dims
        self.n_encoding_dims = n_encoding_dims if n_encoding_dims else 4*n_dims
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

        self.nn = None

    def _init_nn(self):
        self.nn = DNN(self.n_dims, self.n_states, self.n_encoding_dims)

    def log_likelihood(self, obs):
        self._check_obs(obs)
        log_likelihood, _ = self.forward(obs)
        return log_likelihood

    def em_step(self, obs, mask = None): 
        self._check_obs(obs)
        if self.nn is None:
            print("Initializing mu...")
            self._init_nn()
        log_likelihood, _ = self.e_step(obs)
        self.m_step(obs, mask)
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

    def m_step(self, obs, mask = None):

        # Prior is the gamma along the time axis at 0
        n_batches, n_observations = self._get_batches_observations(obs)

        if not self.deterministic_start:
            self.log_prior = np.logaddexp.reduce((self.gamma[:, :, 0] - np.log(n_batches))) # Take average of exponential of log probabilities
            self.prior = np.exp(self.log_prior)

        # Stable representation of gamma. Weighted averages are invariant to scaling of probabilities
        gamma_s = np.exp(self.gamma - np.max(self.gamma))

        criterion = nn.KLDivLoss(reduction='batchmean')
        optimizer = torch.optim.Adam(self.nn.parameters(), lr=0.01)
        if mask is not None:
            self._check_mask(mask)
            mask = torch.tensor(mask[:, :, 0]).unsqueeze(1).expand((n_batches, self.n_states, n_observations))
        else:
            mask = torch.full((n_batches, self.n_states, n_observations), True)

        for epoch in range(20):
            optimizer.zero_grad()
            emission_log_likelihood = self._get_emission_log_likelihood(obs, tensor = True)
            loss = criterion(emission_log_likelihood[mask], torch.tensor(gamma_s).float()[mask])
            loss.backward()
            optimizer.step()
            
        # Transition matrix is the expected number of transitions from state i to state j
        for i in range(self.n_states):
            for j in range(self.n_states):
                self.log_A[i, j] = np.logaddexp.reduce(self.xi[:, i, j, :].flatten()) - np.logaddexp.reduce(self.gamma[:, i, :-1].flatten())

        # Scale transition
        self.log_A = self.log_A - np.max(self.log_A)
        self.A = np.exp(self.log_A)
        self.A = self._stochasticize(self.A)
        self.log_A = np.log(self.A)
        return

    def _get_emission_log_likelihood(self, obs, tensor = False):
        output = self.nn(torch.tensor(obs).float()).permute(0,2,1)
        if tensor:
            return output
        else:
            return output.detach().numpy()

    def forward(self, obs, emission_log_likelihood = None):
        if emission_log_likelihood is None:
            emission_log_likelihood = self._get_emission_log_likelihood(obs)
        alpha = np.zeros(emission_log_likelihood.shape)
        _, n_observations = self._get_batches_observations(obs)

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
        _, n_observations = self._get_batches_observations(obs)

        # Last time step is equal to (log) one as there are no future observations
        beta[:, :, -1] = np.zeros(self.n_states)

        for t in range(n_observations - 1)[::-1]:
            # Previous time steps are equal to sum of transitions from state times likelihood
            # Add log probabilitites of transitions and future beta before converting to normal space to avoid numerical underflow
            beta[:, :, t] = np.logaddexp.reduce(beta[:, np.newaxis, :, t+1] + emission_log_likelihood[:, np.newaxis, :, t+1] + self.log_A, axis=2) # Insert additional axis to broadcast along the rows of the transition matrix
            
        return beta
    
    def _normalize(self, x):
        # Ensure sum to 1
        total = np.sum(x)
        if total == 0:
            print("Warning: total is zero. Division by zero encountered.")
            return x
        else:
            return x / total
    
    def _stochasticize(self, x):
        # Ensure rows sum to 1
        row_sums = np.sum(x, axis=1)[:, np.newaxis]
        row_sums[row_sums == 0] = 1  # replace 0s with 1s to avoid division by zero
        return x / row_sums

    def _get_batches_observations(self, obs):
        return obs.shape[0], obs.shape[1]

    def _check_obs(self, obs):
        assert len(obs.shape) == 3, "obs must have three dimensions"
        assert obs.shape[2] == self.n_dims, f"Third dimension of obs must be equal to {self.n_dims}"

    def _check_mask(self, mask):
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                assert np.all(mask[i, j, :]) or not np.any(mask[i, j, :]), "Third dimension is not all Trues or all Falses"
