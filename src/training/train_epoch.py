import torch
from sklearn.metrics import confusion_matrix

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nRunning on {device}")


def train_epoch(epoch, data_loader, model, criterion, optimizer, epoch_logger):
    print('Epoch: {}, Learning Rate: {}'.format(epoch, optimizer.param_groups[0]['lr']))

    # Switch to train mode
    model.train()
    train_loss, total, correct, confusion_vector = 0, 0, 0, 0

    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs = inputs.to(device)

        # Compute output
        outputs = model(inputs)
        loss = criterion(outputs.cpu(), targets.unsqueeze(1).float())
        # Calculate loss
        train_loss += loss.item()
        # Make predictions
        predicted = outputs >= 0.5
        total += targets.size(0)
        # Extract evaluation metrics
        confusion_vector += confusion_matrix(targets.cpu(), predicted.cpu(), labels=[0, 1])

        # Compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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

    # Save model's checkpoints
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    epoch_logger.log({
        'epoch': epoch,
        'loss': train_loss / (batch_idx + 1),
        'lr': optimizer.param_groups[0]['lr'],
        'accuracy': 100. * ((tp + tn) / total),
        'balanced_accuracy': 100. * (recall + specificity) / 2,
        'recall': recall,
        'precision': precision,
        'F1-score': F1_score
    })

    print(f"Training Balanced Accuracy: {100. * (recall + specificity) / 2}")

    return state
