import torch
from sklearn.metrics import confusion_matrix

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def val_epoch(epoch, data_loader, model, criterion, scheduler, epoch_logger):
    # Switch to evaluation mode
    model.eval()
    val_loss, total, confusion_vector = 0, 0, 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):

            inputs = inputs.to(device)

            # Compute output
            outputs = model(inputs)
            loss = criterion(outputs.cpu(), targets.unsqueeze(1).float())
            # Calculate loss
            val_loss += loss.item()
            # Make predictions
            predicted = outputs >= 0.5
            total += targets.size(0)
            # Extract evaluation metrics
            confusion_vector += confusion_matrix(targets.cpu(), predicted.cpu(), labels=[0, 1])

    # Update learning rate scheduler
    if scheduler:
        scheduler.step(val_loss / (batch_idx + 1))

    tn = confusion_vector[0][0]
    tp = confusion_vector[1][1]
    fp = confusion_vector[0][1]
    fn = confusion_vector[1][0]
    recall = max(0, tp / (tp + fn))
    precision = max(0, tp / (tp + fp))
    specificity = max(0, tn / (tn + fp))
    if precision + recall == 0:
        F1_score = 0
    else:
        F1_score = 2 * (recall * precision) / (precision + recall)

    epoch_logger.log({
        'epoch': epoch,
        'loss': val_loss / (batch_idx + 1),
        'accuracy': 100. * ((tp + tn) / total),
        'balanced_accuracy': 100. * (recall + specificity) / 2,
        'recall': recall,
        'precision': precision,
        'F1-score': F1_score
    })

    print(f"Validation Balanced Accuracy: {100. * (recall + specificity) / 2}")
    print()

    return scheduler, val_loss / (batch_idx + 1)