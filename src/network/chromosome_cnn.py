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


# 1. Residual Block: add the initial information after 1 conv layer to avoid gradient going to 0, I also applied dilation to learn better long term features
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, dilation=4):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=(kernel_size//2) * dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = out + identity
        return F.relu(out)


# 2. Squeeze-and-Excitation (SE) Block: lightest attention mechanism i was able to find
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch, channels, _ = x.shape
        y = self.global_avg_pool(x).view(batch, channels)
        y = F.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y))
        y = y.view(batch, channels, 1)
        return x * y


# 3. Multi-Scale Convolution Block: multi attention model that should find longer range motifs (it runs in parallel 2 conv, one with long and one with short kernel)
class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConv, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding="same")
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size=25, padding="same")

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        
        return x1 + x2


# 4. Full CNN Model with all modifications
class ModifiedChromosomeCNN(nn.Module):
    def __init__(self, input_dim, seq_len, output_dim):
        super(ModifiedChromosomeCNN, self).__init__()

        self.input_dim = input_dim
        self.seq_len = seq_len

        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5, padding=2)

        self.multi_scale = MultiScaleConv(64, 128)

        self.resblock1 = ResidualBlock(128, 128, dilation=4)

        self.se_block = SEBlock(128)

        self.fc1 = None
        self.fc2 = nn.Linear(128, 1)

    def initialize_fc1(self, x):
        if self.fc1 is None:
            flattened_size = x.shape[1] * x.shape[2]
            self.fc1 = nn.Linear(flattened_size, 128).to(x.device)

    def forward(self, inputs_seq):
        x = inputs_seq

        x = F.relu(self.conv1(x))
        x = self.multi_scale(x)
        x = self.resblock1(x)
        x = self.se_block(x)

        x = flatten(x, start_dim=1)

        if self.fc1 is None:
            fc1_input_size = x.shape[1]
            self.fc1 = nn.Linear(fc1_input_size, 128).to(x.device)

        x = self.fc1(x)
        x = self.fc2(x)

        return x
