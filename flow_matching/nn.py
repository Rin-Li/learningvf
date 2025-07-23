import torch
from torch import nn, Tensor
from flow_matching.helpers import SinusoidalPosEmb

class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor: 
        return torch.sigmoid(x) * x

# Model class
class MLP(nn.Module):
    def __init__(self, input_dim: int = 32, time_dim: int = 1, hidden_dim: int = 128):
        super().__init__()
        
        self.input_dim = input_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim
        self.time_embedding_dim = 32
        
        self.time_step_encoder = nn.Sequential(
            SinusoidalPosEmb(self.time_embedding_dim),
            nn.Linear(self.time_embedding_dim, self.time_embedding_dim * 4),
            nn.Mish(),
            nn.Linear(self.time_embedding_dim * 4, self.time_embedding_dim),
        )

        self.main = nn.Sequential(
            nn.Linear(input_dim + self.time_embedding_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        sz = x.size()
        x = x.reshape(-1, self.input_dim)
        t = t.reshape(-1, self.time_dim).float()
        
        t_embed = self.time_step_encoder(t).squeeze(-2)
        h = torch.cat([x, t_embed], dim=1)
        output = self.main(h)
        
        return output.reshape(*sz)


# Test 
if __name__ == "__main__":
    input_dim = 32
    time_dim = 1
    hidden_dim = 128

    model = MLP(input_dim=input_dim, time_dim=time_dim, hidden_dim=hidden_dim)


    x = torch.randn(8, input_dim)  
    t = torch.randn(8, time_dim) 

    # Forward pass
    output = model(x, t)
    print("Output shape:", output.shape)