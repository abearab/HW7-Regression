import numpy as np
import matplotlib.pyplot as plt

# Base class for generic regressor
# (this is already complete!)
class BaseRegressor():

    def __init__(self, num_feats, learning_rate=0.01, tol=0.001, max_iter=100, batch_size=10):

        # Weights are randomly initialized
        self.W = np.random.randn(num_feats + 1).flatten()

        # Store hyperparameters
        self.lr = learning_rate
        self.tol = tol
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.num_feats = num_feats

        # Define empty lists to store losses over training
        self.loss_hist_train = []
        self.loss_hist_val = []
    
    def make_prediction(self, X):
        raise NotImplementedError
    
    def loss_function(self, y_true, y_pred):
        raise NotImplementedError
        
    def calculate_gradient(self, y_true, X):
        raise NotImplementedError
    
    def train_model(self, X_train, y_train, X_val, y_val):

        # Padding data with vector of ones for bias term
        X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
        X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    
        # Defining intitial values for while loop
        prev_update_size = 1
        iteration = 1

        # Repeat until convergence or maximum iterations reached
        while prev_update_size > self.tol and iteration < self.max_iter:

            # Shuffling the training data for each epoch of training
            shuffle_arr = np.concatenate([X_train, np.expand_dims(y_train, 1)], axis=1)
            np.random.shuffle(shuffle_arr)
            X_train = shuffle_arr[:, :-1]
            y_train = shuffle_arr[:, -1].flatten()

            # Create batches
            num_batches = int(X_train.shape[0] / self.batch_size) + 1
            X_batch = np.array_split(X_train, num_batches)
            y_batch = np.array_split(y_train, num_batches)

            # Create list to save the parameter update sizes for each batch
            update_sizes = []

            # Iterate through batches (one of these loops is one epoch of training)
            for X_train, y_train in zip(X_batch, y_batch):

                # Make prediction and calculate loss
                y_pred = self.make_prediction(X_train)
                train_loss = self.loss_function(y_train, y_pred)
                self.loss_hist_train.append(train_loss)

                # Update weights
                prev_W = self.W
                grad = self.calculate_gradient(y_train, X_train)
                new_W = prev_W - self.lr * grad 
                self.W = new_W

                # Save parameter update size
                update_sizes.append(np.abs(new_W - prev_W))

                # Compute validation loss
                val_loss = self.loss_function(y_val, self.make_prediction(X_val))
                self.loss_hist_val.append(val_loss)

            # Define step size as the average parameter update over the past epoch
            prev_update_size = np.mean(np.array(update_sizes))

            # Update iteration
            iteration += 1
    
    def plot_loss_history(self):

        # Make sure training has been run
        assert len(self.loss_hist_train) > 0, "Need to run training before plotting loss history."

        # Create plot
        fig, axs = plt.subplots(2, figsize=(8, 8))
        fig.suptitle('Loss History')
        axs[0].plot(np.arange(len(self.loss_hist_train)), self.loss_hist_train)
        axs[0].set_title('Training')
        axs[1].plot(np.arange(len(self.loss_hist_val)), self.loss_hist_val)
        axs[1].set_title('Validation')
        plt.xlabel('Steps')
        fig.tight_layout()
        plt.show()

    def reset_model(self):
        self.W = np.random.randn(self.num_feats + 1).flatten()
        self.loss_hist_train = []
        self.loss_hist_val = []
        
# Implement logistic regression as a subclass
class LogisticRegressor(BaseRegressor):

    def __init__(self, num_feats, learning_rate=0.01, tol=0.001, max_iter=100, batch_size=10):
        super().__init__(
            num_feats,
            learning_rate=learning_rate,
            tol=tol,
            max_iter=max_iter,
            batch_size=batch_size
        )
    
    def make_prediction(self, X) -> np.array:
        """
        TODO: Implement logistic function to get estimates (y_pred) for input X values. The logistic
        function is a transformation of the linear model into an "S-shaped" curve that can be used
        for binary classification.

        Arguments: 
            X (np.ndarray): Matrix of feature values.

        Returns: 
            The predicted labels (y_pred) for given X.
        """
        # Add bias term if not already present
        if X.shape[1] != self.num_feats + 1 or X.shape[1] == self.num_feats:
            X = np.hstack([X, np.ones((X.shape[0], 1))])
        
        # Calculate linear model
        linear_model = np.dot(X, self.W)
        
        # Apply logistic function – this is the sigmoid function
        y_pred = 1 / (1 + np.exp(-linear_model))
        
        return y_pred
    
    def loss_function(self, y_true, y_pred) -> float:
        """
        TODO: Implement binary cross entropy loss, which assumes that the true labels are either
        0 or 1. (This can be extended to more than two classes, but here we have just two.)

        Arguments:
            y_true (np.array): True labels.
            y_pred (np.array): Predicted labels.

        Returns: 
            The mean loss (a single number).
        """
        # clip the predicted values to avoid log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        # calculate the mean loss
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    def calculate_gradient(self, y_true, X) -> np.ndarray:
        """
        TODO: Calculate the gradient of the loss function with respect to the given data. This
        will be used to update the weights during training.

        Arguments:
            y_true (np.array): True labels.
            X (np.ndarray): Matrix of feature values.

        Returns: 
            Vector of gradients.
        """
        # Add bias term if not already present
        if X.shape[1] != self.num_feats + 1 or X.shape[1] == self.num_feats:
            X = np.hstack([X, np.ones((X.shape[0], 1))])
        
        # Calculate the predicted values
        y_pred = self.make_prediction(X)
        
        # Calculate the gradient of the loss function
        gradient = np.dot(X.T, (y_pred - y_true)) / y_true.size
        
        return gradient

    def hyperparameter_tuning(self, 
                              X_train, y_train, X_val, y_val, 
                              learning_rates=[10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1],
                              batch_sizes=[10, 50, 100],
                              max_iters=[25, 50, 100, 250, 500, 1000]
                              ):
        """
        An optional method for hyperparameter tuning. This is not required for the assignment,
        but can be useful for finding the best hyperparameters for the model.
        """
        # Initialize best hyperparameters and best loss
        best_lr = None
        best_bs = None
        best_loss = float('inf')

        # Iterate over all combinations of hyperparameters
        for lr in learning_rates:
            for bs in batch_sizes:
                # Reset model
                self.reset_model()
                # Train model with current hyperparameters
                self.lr = lr
                self.batch_size = bs
                for mi in max_iters:
                    self.max_iter = mi
                    self.train_model(X_train, y_train, X_val, y_val)
                    # Calculate validation loss
                    val_loss = self.loss_function(y_val, self.make_prediction(X_val))
                    # Update best hyperparameters if current loss is lower
                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_lr = lr
                        best_bs = bs
                        best_mi = mi

        # Set the best hyperparameters
        self.lr = best_lr
        self.batch_size = best_bs
        self.max_iter = best_mi
        print(f"Best hyperparameters: learning rate = {best_lr}, batch size = {best_bs}, max iterations = {best_mi}")