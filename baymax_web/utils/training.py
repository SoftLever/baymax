from pneumonia_cnn import lung_sound_classifier_model, device
from dataset_dry_run import train_dl
import torch
import pandas as pd

# ----------------------------
# Training Loop
# ----------------------------
def training(model, train_dl, num_epochs):
    # Load previous training results
    pt = pd.read_csv("training.csv", sep=",", header=0)

    # Loss Function, Optimizer and Scheduler
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.0001,
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
            # inputs = data[0].to(device)
            # labels = data[1].unsqueeze(1).float().to(device)
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
        
        # Print stats at the end of the epoch
        num_batches = len(train_dl)
        avg_loss = running_loss / num_batches
        acc = correct_prediction/total_prediction
        print(f'Epoch: {epoch + 1}, Loss: {avg_loss}, Accuracy: {acc}')
        
        # Add current training results
        pt.loc[len(pt.index)] = [avg_loss, acc]

        # Save if we have an improved model
        # A model with a lower loss is better perfoming
        if avg_loss <= pt['avg_loss'].min():
            print("Saving model")
            torch.save(model.state_dict(), "/home/collins/Desktop/projects/baymax/baymax_web/inference/saved_model")

    # Save current training results to CSV
    pt.to_csv("training.csv", index=False)

    print('Finished Training')
  
# Load saved model
# path = '/home/collins/Desktop/projects/baymax/saved_model'
# lung_sound_classifier_model.load_state_dict(torch.load(path))

num_epochs=20
training(lung_sound_classifier_model, train_dl, num_epochs)

