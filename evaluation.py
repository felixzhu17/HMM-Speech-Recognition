import numpy as np
import torch
from hmmlearn import hmm
from torch.utils.data import DataLoader


def generate_HMM_samples(
    n_samples=100,
    n_batches=100,
    n_components=2,
    n_features=3,
    startprob=None,
    transmat=None,
    means=None,
    covs=None,
):
    """
    Generate samples from a Gaussian Hidden Markov Model.

    Args:
        n_samples: Number of samples to generate for each batch.
        n_batches: Number of batches of samples to generate.
        n_components: Number of hidden states in the HMM.
        n_features: Dimensionality of the observations.
        startprob: Starting probabilities for each hidden state.
        transmat: Transition probabilities between hidden states.
        means: Means of each hidden state.
        covs: Covariances of each hidden state.

    Returns:
        X: Generated samples. Shape: (n_batches, n_samples, n_features)
    """
    # Create a Gaussian HMM
    model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag")

    # Manually specify the model parameters
    model.startprob_ = np.array([1, 0]) if startprob is None else np.array(startprob)

    model.transmat_ = (
        np.array(
            [
                [0.7, 0.3],
                [0.4, 0.6],
            ]
        )
        if transmat is None
        else np.array(transmat)
    )

    # Means of each hidden state
    model.means_ = (
        np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]])
        if means is None
        else np.array(means)
    )

    # Covariances of each hidden state
    model.covars_ = (
        np.ones((n_components, n_features)) if covs is None else np.array(covs)
    )

    # Generate samples
    X = np.stack([model.sample(n_samples)[0] for _ in range(n_batches)])

    return X


def top1_accuracy(output, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        pred = torch.argmax(output, 1)
        correct = pred.eq(target.view_as(pred))
        correct = correct.float().sum()
        return correct / len(target)


def topk_accuracy(output, target, k=3):
    """Computes the topk accuracy"""
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        correct = pred.eq(target.view(-1, 1).expand_as(pred))
        correct_k = correct.view(-1).float().sum()
        return correct_k / len(target)


def compute_nn_accuracies(model, test_data, batch_size=1):
    """
    Computes the top-1 and top-3 accuracy for a model on given data.

    Args:
        model (torch.nn.Module): The model to evaluate.
        test_data (torch.utils.data.Dataset): The data to evaluate on.

    Returns:
        float: The top-1 accuracy.
        float: The top-3 accuracy.
    """
    # Ensure model is in evaluation mode
    model.eval()

    top1_acc = 0
    top3_acc = 0
    total_samples = 0

    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    with torch.no_grad():
        for i, (inputs, targets, masks) in enumerate(test_dataloader):
            outputs = model(inputs)
            top1_acc += top1_accuracy(outputs, targets) * len(targets)
            top3_acc += topk_accuracy(outputs, targets) * len(targets)
            total_samples += len(targets)

    # Calculate the average accuracies
    top1_acc = top1_acc / total_samples
    top3_acc = top3_acc / total_samples

    return top1_acc.item(), top3_acc.item()


def calculate_hmm_accuracy(test_hmm, hmm_models):
    """
    Calculate the top-1 and top-3 accuracy for a set of HMM models on given test data.

    Args:
        test_hmm (tuple): A tuple of two lists, first containing the test samples and second containing the corresponding labels.
        hmm_models (dict): A dictionary of trained HMM models. The keys are the labels (words), and the values are the corresponding models.

    Returns:
        float: The top-1 accuracy, which is the percentage of test samples for which the highest likelihood model matches the true label.
        float: The top-3 accuracy, which is the percentage of test samples for which the true label is among the three models with highest likelihood.
    """
    top3_correct = 0
    top1_correct = 0
    total_samples = 0

    for test_sample, test_label in zip(*test_hmm):
        total_samples += 1
        log_likelihoods = {}
        for word in hmm_models.keys():
            log_likelihoods[word] = hmm_models[word].log_likelihood(
                test_sample[np.newaxis]
            )

        # Sort the words based on their log likelihoods in descending order
        sorted_words = sorted(log_likelihoods, key=log_likelihoods.get, reverse=True)

        # Check if the actual label is in the top-k predictions
        if test_label in sorted_words[:3]:
            top3_correct += 1

        # Check if the actual label is the top-1 prediction
        if test_label == sorted_words[0]:
            top1_correct += 1

    # Calculate accuracies
    top3_accuracy = top3_correct / total_samples
    top1_accuracy = top1_correct / total_samples

    return top1_accuracy, top3_accuracy
