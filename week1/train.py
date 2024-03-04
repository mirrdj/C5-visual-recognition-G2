import torch

def train(model, dataloader_train, criterion, optimizer, params, device):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in dataloader_train:
        imgs, labels = imgs.to(device), labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(imgs, params)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Compute training accuracy and loss
        train_loss += loss.item() * imgs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss /= len(dataloader_train.dataset)
    train_accuracy = correct/total

    return train_loss, train_accuracy


def validation(model, dataloader_val, criterion, params, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in dataloader_val:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs =  model(imgs, params)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(dataloader_val)
    val_accuracy = correct / total

    return val_loss, val_accuracy