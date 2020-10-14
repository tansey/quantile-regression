'''
A basic NN that optimizes mean squared error.
'''
import numpy as np
import os
import sys
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from utils import create_folds, batches
from torch_utils import clip_gradient, logsumexp


'''Neural network to map from X to expectation of y.'''
class SqErrNetworkModule(nn.Module):
    def __init__(self, X_means, X_stds, y_mean, y_std, n_out=1):
        super(SqErrNetworkModule, self).__init__()
        self.X_means = X_means
        self.X_stds = X_stds
        self.y_mean = y_mean
        self.y_std = y_std
        self.n_in = X_means.shape[1]
        self.n_out = n_out
        self.fc_in = nn.Sequential(
                nn.Linear(self.n_in, 200),
                nn.Dropout(0.1),
                nn.ReLU(),
                nn.BatchNorm1d(200),
                nn.Linear(200, 200),
                nn.Dropout(0.1),
                nn.ReLU(),
                nn.BatchNorm1d(200),
                nn.Linear(200, self.n_out if len(self.y_mean.shape) == 1 else self.n_out * self.y_mean.shape[1]))
        
        
    def forward(self, x):
        fout = self.fc_in(x)

        # If we are dealing with multivariate responses, reshape to the (d x q) dimensions
        if len(self.y_mean.shape) != 1:
            fout = fout.reshape((-1, self.y_mean.shape[1], self.n_out))

        return fout

    def predict(self, X):
        self.eval()
        self.zero_grad()
        tX = autograd.Variable(torch.FloatTensor((X - self.X_means) / self.X_stds), requires_grad=False)
        fout = self.forward(tX)
        z = fout.data.numpy() * self.y_std[...,None] + self.y_mean[...,None]
        return z

class SqErrNetwork:
    def __init__(self):
        self.label = 'SqErr Network'
        self.filename = 'sqerr'

    def fit(self, X, y):
        self.model = fit_nn(X, y)

    def predict(self, X):
        return self.model.predict(X)

def fit_nn(X, y, 
            nepochs=100, val_pct=0.1,
            batch_size=None, target_batch_pct=0.01,
            min_batch_size=20, max_batch_size=100,
            verbose=False, lr=1e-1, weight_decay=0.0, patience=5,
            init_model=None, splits=None, file_checkpoints=True,
            clip_gradients=False, **kwargs):
    if file_checkpoints:
        import uuid
        tmp_file = '/tmp/tmp_file_' + str(uuid.uuid4())

    if batch_size is None:
        batch_size = min(X.shape[0], max(min_batch_size, min(max_batch_size, int(np.round(X.shape[0]*target_batch_pct)))))
        if verbose:
            print('Auto batch size chosen to be {}'.format(batch_size))

    # Standardize the features and response (helps with gradient propagation)
    Xmean = X.mean(axis=0, keepdims=True)
    Xstd = X.std(axis=0, keepdims=True)
    Xstd[Xstd == 0] = 1 # Handle constant features
    ymean, ystd = y.mean(axis=0, keepdims=True), y.std(axis=0, keepdims=True)
    tX = autograd.Variable(torch.FloatTensor((X - Xmean) / Xstd), requires_grad=False)
    tY = autograd.Variable(torch.FloatTensor((y - ymean) / ystd), requires_grad=False)

    # Create train/validate splits
    if splits is None:
        indices = np.arange(X.shape[0], dtype=int)
        np.random.shuffle(indices)
        train_cutoff = int(np.round(len(indices)*(1-val_pct)))
        train_indices = indices[:train_cutoff]
        validate_indices = indices[train_cutoff:]
    else:
        train_indices, validate_indices = splits

    # Initialize the model
    model = SqErrNetworkModule(Xmean, Xstd, ymean, ystd) if init_model is None else init_model

    # Save the model to file
    if file_checkpoints:
        torch.save(model, tmp_file)
    else:
        import pickle
        model_str = pickle.dumps(model)

    # Setup the SGD method
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, nesterov=True, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    # Track progress
    train_losses, val_losses, best_loss = np.zeros(nepochs), np.zeros(nepochs), None
    num_bad_epochs = 0

    if verbose:
        print('ymax and min:', tY.max(), tY.min())

    # Create the squared error loss function
    def sqerr_loss(yhat, tidx):
        if len(yhat.shape) > 1:
            return (tY[tidx,None,None] - yhat)**2
        return (tY[tidx,None] - yhat)**2

    # Train the model
    for epoch in range(nepochs):
        if verbose:
            print('\t\tEpoch {}'.format(epoch+1))
            sys.stdout.flush()

        # Track the loss curves
        train_loss = torch.Tensor([0])
        for batch_idx, batch in enumerate(batches(train_indices, batch_size, shuffle=True)):
            if verbose and (batch_idx % 100 == 0):
                print('\t\t\tBatch {}'.format(batch_idx))
            tidx = autograd.Variable(torch.LongTensor(batch), requires_grad=False)

            # Set the model to training mode
            model.train()

            # Reset the gradient
            model.zero_grad()

            # Run the model and get the predicted mean
            yhat = model(tX[tidx])
            
            # Loss for optimizing
            loss = sqerr_loss(yhat, tidx).mean()

            # Calculate gradients
            loss.backward()

            # Clip the gradients
            if clip_gradients:
                clip_gradient(model)

            # Apply the update
            # [p for p in model.parameters() if p.requires_grad]
            optimizer.step()

            # Track the loss
            train_loss += loss.data

            if np.isnan(loss.data.numpy()):
                import warnings
                warnings.warn('NaNs encountered in training model.')
                break

        validate_loss = torch.Tensor([0])
        for batch_idx, batch in enumerate(batches(validate_indices, batch_size, shuffle=False)):
            if verbose and (batch_idx % 100 == 0):
                print('\t\t\tValidation Batch {}'.format(batch_idx))
            tidx = autograd.Variable(torch.LongTensor(batch), requires_grad=False)

            # Set the model to test mode
            model.eval()

            # Reset the gradient
            model.zero_grad()

            # Run the model and get the conditional mixture weights
            yhat = model(tX[tidx])

            # Track the loss
            validate_loss += sqerr_loss(yhat, tidx).sum()

        train_losses[epoch] = train_loss.data.numpy() / float(len(train_indices))
        val_losses[epoch] = validate_loss.data.numpy() / float(len(validate_indices))

        # Adjust the learning rate down if the validation performance is bad
        if num_bad_epochs > patience:
            if verbose:
                print('Decreasing learning rate to {}'.format(lr*0.5))
            scheduler.step(val_losses[epoch])
            lr *= 0.5
            num_bad_epochs = 0

        # If the model blew up and gave us NaNs, adjust the learning rate down and restart
        if np.isnan(val_losses[epoch]):
            if verbose:
                print('Network went to NaN. Readjusting learning rate down by 50%')
            if file_checkpoints:
                os.remove(tmp_file)
            return fit_nn(X, y,
                        nepochs=nepochs, val_pct=val_pct,
                        batch_size=batch_size, target_batch_pct=target_batch_pct,
                        min_batch_size=min_batch_size, max_batch_size=max_batch_size,
                        verbose=verbose, lr=lr*0.5, weight_decay=weight_decay, patience=patience,
                        init_model=init_model, splits=splits, file_checkpoints=file_checkpoints,  **kwargs)

        # Check if we are currently have the best held-out log-likelihood
        if epoch == 0 or val_losses[epoch] <= best_loss:
            if verbose:
                print('\t\t\tSaving test set results.      <----- New high water mark on epoch {}'.format(epoch+1))
            # If so, use the current model on the test set
            best_loss = val_losses[epoch]
            if file_checkpoints:
                torch.save(model, tmp_file)
            else:
                import pickle
                model_str = pickle.dumps(model)
        else:
            num_bad_epochs += 1
        
        if verbose:
            print('Validation loss: {} Best: {}'.format(val_losses[epoch], best_loss))

    # Load the best model and clean up the checkpoints
    if file_checkpoints:
        model = torch.load(tmp_file)
        os.remove(tmp_file)
    else:
        import pickle
        model = pickle.loads(model_str)

    # Return the conditional density model that marginalizes out the grid
    return model





