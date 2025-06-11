import numpy as np


class BatchNormalization:
    def __init__(self, d, epsilon=1e-5, momentum=0.9):
        """
        Initialize Batch Normalization.
        :param d: Dimension of the input features.
        :param epsilon: Small constant to avoid division by zero.
        :param momentum: Momentum for running mean and variance.
        """
        self.d = d
        self.epsilon = epsilon
        self.momentum = momentum

        # Learnable parameters
        self.gamma = np.ones(d)  # Scale parameter
        self.beta = np.zeros(d)  # Shift parameter

        # Running statistics (used in inference mode)
        self.running_mean = np.zeros(d)
        self.running_var = np.zeros(d)

    def forward(self, x, training=True):
        """
        Forward pass for Batch Normalization.
        :param x: Input array of shape (N, d).
        :param training: Whether the forward pass is in training mode.
        :return: Normalized output of shape (N, d).
        """
        if training:
            # Compute mean and variance across the batch
            self.batch_mean = np.mean(x, axis=0)
            self.batch_var = np.var(x, axis=0)

            # Normalize
            self.x_normalized = (x - self.batch_mean) / np.sqrt(self.batch_var + self.epsilon)
            out = self.gamma * self.x_normalized + self.beta

            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.batch_var
        else:
            # Use running statistics in inference mode
            x_normalized = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            out = self.gamma * x_normalized + self.beta

        return out

    def backward(self, dout):
        """
        Backward pass for Batch Normalization.
        :param dout: Upstream gradients of shape (N, d).
        :return: Gradient w.r.t input x of shape (N, d).
        """
        N, d = dout.shape

        # Gradients of scale and shift parameters
        self.dgamma = np.sum(dout * self.x_normalized, axis=0)
        self.dbeta = np.sum(dout, axis=0)

        # Gradient of the normalized input
        dx_normalized = dout * self.gamma

        # Gradient w.r.t variance
        dvar = np.sum(dx_normalized * (self.batch_mean - dout) * -0.5 * (self.batch_var + self.epsilon) ** -1.5, axis=0)

        # Gradient w.r.t mean
        dmean = np.sum(dx_normalized * -1 / np.sqrt(self.batch_var + self.epsilon), axis=0)
        dmean += dvar * np.sum(-2 * (self.x_normalized), axis=0) / N

        # Gradient w.r.t input
        dx = dx_normalized / np.sqrt(self.batch_var + self.epsilon)
        dx += dvar * 2 * (self.batch_mean - dout) / N
        dx += dmean / N

        return dx


class LayerNorm:
    def __init__(self, d, epsilon=1e-5):

        """
        Initialize Layer Normalization with parameters. (Automatic generated)
        :param d: Dimension of the input features.
        :param epsilon: Small constant to avoid division by zero.
        """
        self.d = d
        self.epsilon = epsilon
        self.Gamma = np.ones(d)  # Scale parameter
        self.Beta = np.zeros(d)  # Shift parameter

    def __call__(self, U_t):
        return self.forward(U_t)

    def forward(self, U_t):
        """
        Forward pass for Layer Normalization.
        :param x: Input array of shape (N, d).
        :return: Normalized output of shape (N, d).
        """
        self.mean = np.mean(U_t, axis=1, keepdims=True)  # Mean across features
        self.var = np.var(U_t, axis=1, keepdims=True)    # Variance across features
        self.U_t_norm = (U_t - self.mean) / np.sqrt(self.var + self.epsilon)  # Normalize
        out = self.Gamma * self.U_t_norm + self.Beta  # Scale and shift
        return out

    def backward(self, grad_0):
        """
        Backward pass for Layer Normalization.
        :param grad_0: Upstream gradients of shape (N, d).
        :return: Gradient w.r.t input x of shape (N, d).
        """
        N, d = grad_0.shape

        # Gradients of scale and shift parameters
        self.Gamma_grad = np.sum(grad_0 * self.U_t_norm, axis=0)
        self.Beta_grad = np.sum(grad_0, axis=0)

        # Gradient of the normalized input
        grad_0_norm = grad_0 * self.Gamma

        # Gradient w.r.t variance
        dvar = np.sum(grad_0_norm * (self.U_t_norm * -0.5 / np.sqrt(self.var + self.epsilon)),
                      axis=1, keepdims=True)

        # Gradient w.r.t mean
        dmean = np.sum(grad_0_norm * -1.0 / np.sqrt(self.var + self.epsilon), axis=1, keepdims=True)
        dmean += dvar * np.mean(-2.0 * (self.U_t_norm * np.sqrt(self.var + self.epsilon)), axis=1, keepdims=True)

        # Gradient w.r.t input
        grad_0_norm = grad_0_norm / np.sqrt(self.var + self.epsilon)
        grad_0_norm += dvar * 2.0 * self.U_t_norm * np.sqrt(self.var + self.epsilon) / d
        grad_0_norm += dmean / d

        return grad_0_norm
