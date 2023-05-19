from model import lung_sound_classifier_model, device
from dataset import train_dl
import torch

# ----------------------------
# Training Loop
# ----------------------------
def training(model, train_dl, num_epochs):
    # Loss Function, Optimizer and Scheduler
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                steps_per_epoch=int(len(train_dl)),
                                                epochs=num_epochs,
                                                anneal_strategy='linear')

    # Repeat for each epoch
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0

        # Repeat for each batch in the training set
        for i, data in enumerate(train_dl):
            # Get the input features and target labels, and put them on the GPU
            inputs, labels = data[0].to(device), data[1].to(device)

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Keep stats for Loss and Accuracy
            running_loss += loss.item()

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs,1)
            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

            # if i % 10 == 0:    # print every 10 mini-batches
            #    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
        
        # Print stats at the end of the epoch
        num_batches = len(train_dl)
        avg_loss = running_loss / num_batches
        acc = correct_prediction/total_prediction
        print(f'Epoch: {epoch + 1}, Loss: {avg_loss}, Accuracy: {acc}')

        # Save if we have an improved model
        if avg_loss < 2.08 and acc > 0.63:
            print("Saving model")
            torch.save(model.state_dict(), "/home/collins/Desktop/projects/baymax/saved_model")

    print('Finished Training')
  
# Load saved model
path = '/home/collins/Desktop/projects/baymax/saved_model'
lung_sound_classifier_model.load_state_dict(torch.load(path))

num_epochs=8
training(lung_sound_classifier_model, train_dl, num_epochs)

