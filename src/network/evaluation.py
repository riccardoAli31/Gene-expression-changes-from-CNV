import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.amp import autocast
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.network.chromosome_cnn import ChromosomeCNN

def test_model(model_path, test_loader, total_variables, seq_len, device):

    model = ChromosomeCNN(input_dim=total_variables, seq_len=seq_len, output_dim=1).to(device)
    checkpoint = torch.load(model_path)
    
    input_tensor = torch.zeros(1, model.input_dim, model.seq_len).to(device)
    model(input_tensor)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()

    criterion = nn.BCEWithLogitsLoss()
    test_losses = []
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for stacked_inputs_batch, y_batch in test_loader:
            stacked_inputs_batch = stacked_inputs_batch.to(device)
            y_batch = y_batch.to(device, non_blocking=True)
            #stacked_inputs_batch = stacked_inputs_batch.unsqueeze(0)

            with autocast():
                outputs = model(stacked_inputs_batch)
                loss = criterion(outputs, y_batch)
                test_losses.append(loss.item())

                all_predictions.append(outputs.cpu().numpy())
                all_labels.append(y_batch.cpu().numpy())

    avg_test_loss = sum(test_losses) / len(test_losses)
    print(f"Test MSE: {avg_test_loss:.4f}")

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    probabilities = 1 / (1 + np.exp(-all_predictions))  # Sigmoid function
    predicted_classes = (probabilities >= 0.5).astype(int)  # Convert to 0 or 1 based on threshold

    # Compute accuracy and other metrics
    accuracy = accuracy_score(all_labels, predicted_classes)
    precision = precision_score(all_labels, predicted_classes)
    recall = recall_score(all_labels, predicted_classes)
    f1 = f1_score(all_labels, predicted_classes)
    auc = roc_auc_score(all_labels, probabilities)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'AUC: {auc:.4f}')
    
    return avg_test_loss


def evaluate_model(model, dataset, device):
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
