from torch import nn, flatten
import torch.nn.functional as F

class ChromosomeCNN(nn.Module):
    def __init__(self,  input_dim, seq_len, output_dim):
        super(ChromosomeCNN, self).__init__()
        
        self.input_dim = input_dim
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)

        self.fc1 = None 
        self.fc2 = nn.Linear(128, 1)
        
        self.seq_len = seq_len
        
    def initialize_fc1(self, x):

        if self.fc1 is None: 
            flattened_size = x.shape[1] * x.shape[2] 
            self.fc1 = nn.Linear(flattened_size, 128).to(x.device)     
    
    def forward(self, inputs_seq):
        
        #print(f"Shape of inputs before permute: {inputs_seq.shape}")
        
        x = inputs_seq#.permute(0, 2, 1)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = flatten(x, start_dim=1)
        
        if self.fc1 is None:

            fc1_input_size = x.shape[1]
            self.fc1 = nn.Linear(fc1_input_size, 128)
                
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x