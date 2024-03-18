import torch


class Trainer:
    def train(self, model, dataloader_train, criterion, optimizer, params, device, miner=False):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for imgs, labels in dataloader_train:
            imgs, labels = imgs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            # print(imgs, params)
            outputs = model(imgs)

            if miner:
                indices_tuple = miner(outputs, labels)
                loss = criterion(outputs, labels, indices_tuple)
            else: 
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # if miner:
            #     print("Loss = {}, Number of mined triplets = {}".format(
            #     loss, miner.num_triplets
            #         )
            #     )

            # Compute training accuracy and loss
            train_loss += loss.item() * imgs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss /= len(dataloader_train.dataset)
        train_accuracy = correct/total
        return train_loss, train_accuracy
    

class SiameseTrainer(Trainer):
    def train(self, model, dataloader_train, criterion, optimizer, params, device, miner=False):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for imgs1, imgs2, targets in dataloader_train:
            imgs1, imgs2, targets = imgs1.to(device), imgs2.to(device), targets.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            # print(imgs, params)
            outputs1, outputs2 = model(imgs1, imgs2)
            loss = criterion(outputs1, outputs2, targets)
            loss.backward()
            optimizer.step()

            # print(f"Loss is {loss}")

            # Compute training accuracy and loss
            train_loss += loss.item() * imgs1.size(0)
            _, predicted1 = torch.max(outputs1, 1)
            _, predicted2 = torch.max(outputs2, 1)

            total += targets.size(0)
            correct += ((predicted1 == predicted2) == targets).sum().item()

        train_loss /= len(dataloader_train.dataset)
        train_accuracy = correct / total
        return train_loss, train_accuracy



class TripletTrainer(Trainer):
    def train(self, model, dataloader_train, criterion, optimizer, params, device, miner=False):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for anchor, pos, neg in dataloader_train:
            anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            # print(imgs, params)
            outputs1, outputs2, outputs3 = model(anchor, pos, neg)
            loss = criterion(outputs1, outputs2, outputs3)
            loss.backward()
            optimizer.step()

            print(f"Loss is {loss}")

            # Compute training accuracy and loss
            train_loss += loss.item() * anchor.size(0)
            # _, predicted1 = torch.max(outputs1, 1)
            # _, predicted2 = torch.max(outputs2, 1)
            # _, predicted3 = torch.max(outputs3, 1)

            total += neg.size(0)

            # TODO: Implement correct accuracy according to
            # https://stackoverflow.com/a/47625727

            # def distance(pred1, pred2):
            #     return (pred1 - output1).pow(2).sum(1).sqrt()

            # distance_positive = distance(predicted1, predicted2)
            # distance_negative = distance(predicted1, predicted3)

            # # Check if the network correctly predicts positive closer to anchor than negative
            # if distance_positive < distance_negative:
            #     correct_predictions += 1

            # correct += ((predicted1 == predicted2) == neg).sum().item()

        train_loss /= len(dataloader_train.dataset)
        train_accuracy = correct / total
        return train_loss, 0
