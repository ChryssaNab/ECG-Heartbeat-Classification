import torch
from sklearn.metrics import confusion_matrix

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def test(data_loader, model, criterion):
    # Switch to test mode
    model.eval()
    test_loss, total, confusion_vector = 0, 0, 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):

            inputs = inputs.to(device)

            # Compute output
            outputs = model(inputs)
            loss = criterion(outputs.cpu(), targets.unsqueeze(1).float())
            # Calculate loss
            test_loss += loss.item()
            # Make predictions
            predicted = outputs >= 0.5
            total += targets.size(0)
            # Extract evaluation metrics
            confusion_vector += confusion_matrix(targets.cpu(), predicted.cpu(), labels=[0, 1])

    tn = confusion_vector[0][0]
    tp = confusion_vector[1][1]
    fp = confusion_vector[0][1]
    fn = confusion_vector[1][0]
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    specificity = tn / (tn + fp)

    # Save checkpoints
    state = {
        'test_loss': test_loss / (batch_idx + 1),
        'test_accuracy': 100. * ((tp + tn) / total),
        'test_balanced_accuracy': 100. * (recall + specificity) / 2,
        'test_recall': recall,
        'test_precision': precision,
        'test_F1-score': 2 * (recall * precision) / (precision + recall)
    }

    return state