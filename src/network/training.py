import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm import tqdm
import copy
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


class EarlyStopping:
    def _init_(self, patience=5, delta=0):

        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def _call_(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def create_tqdm_bar(iterable, desc):
    return tqdm(enumerate(iterable),total=len(iterable), ncols=150, desc=desc)

def train_model(model: nn.Module, hparams: dict, train_loader: DataLoader,
                val_loader: DataLoader, tb_logger, device, name='default'):
    """
    Model training function.
    """
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=hparams.get('lr', 1e-3)
    )
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    early_stopping = EarlyStopping()
    epochs = hparams.get('epochs', 3)

    model = model.to(device)

    validation_loss = 0
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

            optimizer.zero_grad()

            with autocast(device_type=device):
                outputs = model(stacked_inputs_batch)
                loss = criterion(outputs, y_batch)
            
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            train_losses.append(loss.item())
            
            # update the progress bar with running average loss
            train_loop.set_postfix(
                train_loss = "{:.8f}".format(train_loss / (train_i + 1)), 
                val_loss = "{:.8f}".format(validation_loss)
                )

            # update the tensorboard logger.
            tb_logger.add_scalar(
                f'CNV_model_{name}/train_loss', loss.item(), 
                epoch * len(train_loader) + train_i
                )
            
        avg_train_loss = sum(train_losses) / len(train_losses)
        train_losses_avg.append(avg_train_loss)

        tb_logger.add_scalar(
            f'CNV_model_{name}/avg_train_loss', avg_train_loss, epoch
            )

        # validation
        model.eval()
        validation_loss = 0
        val_losses = []
        all_val_predictions = []
        all_val_labels = []
        val_loop = create_tqdm_bar(
            val_loader, desc=f'Validation Epoch [{epoch + 1}/{epochs}]'
            )
        with torch.no_grad():
            for val_i, (stacked_inputs_batch, y_batch) in val_loop:

                stacked_inputs_batch = stacked_inputs_batch.to(device)
                y_batch = y_batch.to(device) # , non_blocking=True

                with torch.no_grad(), autocast():
                    y_pred = model(stacked_inputs_batch)
                    loss = criterion(y_pred, y_batch)
                    val_losses.append(loss.item())
                    all_val_predictions.append(y_pred)
                    all_val_labels.append(y_batch)
                validation_loss += loss.item()

                # update the progress bar with running average val loss
                val_loop.set_postfix(
                    val_loss = "{:.8f}".format(validation_loss / (val_i + 1))
                    )

                # update the tensorboard logger.
                tb_logger.add_scalar(
                    f'CNV_model_{name}/val_loss', loss.item(), 
                    epoch * len(val_loader) + val_i
                    )

            avg_val_loss = sum(val_losses) / len(val_losses)
            val_losses_avg.append(avg_val_loss)

            tb_logger.add_scalar(
                f'CNV_model_{name}/avg_val_loss', avg_val_loss, epoch
            )

            # accuracy
            val_accuracy = accuracy_score(all_val_predictions, all_val_labels)
            tb_logger.add_scalar(
                f'CNV_model_{name}/val_acc', val_accuracy, epoch
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model = copy.deepcopy(model.state_dict())

            early_stopping(avg_val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break
        
        # This value is used for the progress bar of the training loop.
        validation_loss /= len(val_loader)

    # plot_df = pd.DataFrame({
    #     'epoch': list(range(len(train_losses_avg[1:]))) * 2,
    #     'avg_train_loss': train_losses_avg[1:],
    #     'avg_val_loss': val_losses_avg[1:]
    #     })
    # p = ggplot(data=plot_df, mapping=aes(x='epoch')) +\
    #     geom_line(aes(y='avg_train_loss', color='Train loss')) +\
    #     geom_line(aes(y='avg_val_loss'), color='Val loss') +\
    #     labs(title='Average Loss during training', x='Epoch', y='Avg. Loss')
    # TODO: ggsave
    p.show()
    plt.plot(train_losses_avg[1:], label='Train Loss')
    plt.plot(val_losses_avg[1:], label='Val Loss')
    plt.legend()
    plt.show()
            
    return avg_val_loss, best_model
