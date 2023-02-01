import torch
from sklearn.metrics import confusion_matrix

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_epoch(epoch, data_loader, model, criterion, optimizer):
    print('\nEpoch: %d' % epoch)
    # Switch to train mode
    model.train()
    train_loss, total, correct, confusion_vector = 0, 0, 0, 0

    for batch_idx, (inputs, targets) in enumerate(data_loader):

        inputs = inputs.to(device)

        # Compute output
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
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
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)

    # Save checkpoints
    state = {
        'net': model.state_dict(),
        'epoch': epoch,
        'train_loss': train_loss / (batch_idx + 1),
        'train_accuracy': 100. * ((tp + tn) / total),
        'train_recall': recall,
        'train_precision': precision,
        'train_F1-score': 2 * (recall * precision) / (precision + recall)
    }

    print(f"Training accuracy: {100. * ((tp + tn) / total)}")

    return state