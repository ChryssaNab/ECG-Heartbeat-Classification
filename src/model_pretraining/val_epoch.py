import torch
from sklearn.metrics import confusion_matrix

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def val_epoch(epoch, data_loader, model, criterion):
    print('\nEpoch: %d' % epoch)
    # Switch to evaluation mode
    model.eval()
    val_loss, total, confusion_vector = 0, 0, 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):

            inputs = inputs.to(device)

            # Compute output
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            # Calculate loss
            val_loss += loss.item()
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

    # Save checkpoints
    state = {
        'epoch': epoch,
        'val_loss': val_loss / (batch_idx + 1),
        'val_accuracy': 100. * ((tp + tn) / total),
        'val_recall': recall,
        'val_precision': precision,
        'val_F1-score': 2 * (recall * precision) / (precision + recall)
    }

    print(f"Validation accuracy: {100. * ((tp + tn) / total)}")

    return state