from torch import nn, optim
from torch.utils.data import DataLoader

def create_tqdm_bar(iterable, desc):
    return tqdm(enumerate(iterable),total=len(iterable), ncols=150, desc=desc)

def train_model(model, train_loader: DataLoader, val_loader: DataLoader, tb_logger, name='default'):
    """
    Model training function.
    """
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hparams.get('lr', 1e-3)
    )
    criterion = nn.BCEWithLogitsLoss()
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    epochs = hparams.get('epochs', 3)

    model = model.to(device)

    validation_loss = 0

    # Riccardo
    best_val_loss = float('inf')
    
    train_losses_avg = []
    val_losses_avg = []
    best_model = None

    for epoch in range(epochs):

        # training
        model.train()

        train_loss = 0
        train_losses = []
        
        train_loop = create_tqdm_bar(
            train_loader, desc=f'Training Epoch [{epoch + 1}/{epochs}]'
            )
        for train_i, (stacked_inputs_batch, y_batch) in train_loop:

            stacked_inputs_batch = stacked_inputs_batch.to(device)
            y_batch = y_batch.to(device) # , non_blocking=True
            #stacked_inputs_batch = stacked_inputs_batch.unsqueeze(0)

            optimizer.zero_grad()

            # with autocast(device_type=device):   
            outputs = model(stacked_inputs_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            # scheduler.step()

            train_loss += loss.item()
            train_losses.append(loss.item())
            
            # Update the progress bar.
            train_loop.set_postfix(
                curr_train_loss = "{:.8f}".format(train_loss / (train_i + 1)), 
                val_loss = "{:.8f}".format(validation_loss)
                )

            # Update the tensorboard logger.
            tb_logger.add_scalar(
                f'CNV_model_{name}/train_loss', loss.item(), 
                epoch * len(train_loader) + train_i
                )
            
        avg_train_loss = sum(train_losses) / len(train_losses)
        train_losses_avg.append(avg_train_loss)
    
        # print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss / len(train_loader):.4f}")
    
        # validation
        model.eval()
        val_losses = []
        val_loss = 0
        val_loop = create_tqdm_bar(
            val_loader, desc=f'Validation Epoch [{epoch + 1}/{epochs}]'
            )
        with torch.no_grad():
            for val_i, (stacked_inputs_batch, y_batch) in val_loop:

                stacked_inputs_batch = stacked_inputs_batch.to(device)
                y_batch = y_batch.to(device) # , non_blocking=True
                #stacked_inputs_batch = stacked_inputs_batch.unsqueeze(0)

                # with torch.no_grad(), autocast():
                y_pred = model(stacked_inputs_batch)
                loss = criterion(y_pred, y_batch)
                val_losses.append(loss.item())
                val_loss += loss.item()

                # Update the progress bar.
                val_loop.set_postfix(
                    val_loss = "{:.8f}".format(validation_loss / (val_i + 1))
                    )

                # Update the tensorboard logger.
                tb_logger.add_scalar(
                    f'CNV_model_{name}/val_loss', loss.item(), 
                    epoch * len(val_loader) + val_i
                    )

            avg_val_loss = sum(val_losses) / len(val_losses)
            val_losses_avg.append(avg_val_loss)
            # print(f'Epoch {epoch+1}, Val loss: {avg_val_loss}')
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model = copy.deepcopy(model.state_dict())
        
        # This value is used for the progress bar of the training loop.
        validation_loss /= len(val_loader)

    plt.plot(train_losses_avg[1:], label='Train Loss')
    plt.plot(val_losses_avg[1:], label='Val Loss')
    plt.legend()
    plt.show()
            
    return avg_val_loss, best_model

def evaluate_model(model, dataset):
    model.eval()
    model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    loss = 0
    for dna_embedding, target in dataloader:
        dna_embedding, target = dna_embedding.to(device), target.to(device)
        y_hat = model(dna_embedding)
        loss += criterion(target, y_hat).item()
    return 1.0 / (2 * (loss / len(dataloader)))