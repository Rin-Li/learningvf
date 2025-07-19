import torch
from matplotlib import patches
import math

class Circle:
    def __init__(self,center,radius,device='cpu'):
        self.center = center.to(device)
        self.radius = radius
        self.device = device

    def signed_distance(self,p):
        # p: N x 2
        N = p.size(0)
        return (torch.norm(p-self.center.unsqueeze(0).expand(N,-1),dim=1) - self.radius).unsqueeze(-1)

    def normal(self,p):
        d = self.signed_distance(p)
        grad = torch.autograd.grad(d.sum(), p, create_graph=True, retain_graph=True)[0]
        n = torch.nn.functional.normalize(grad,dim = -1)
        return n

    def sample_surface(self,N):
        theta = torch.rand(N,1).to(self.device) * 2 * math.pi
        x = torch.cat([torch.cos(theta).to(self.device),torch.sin(theta).to(self.device)],dim=-1)
        return x * self.radius + self.center.unsqueeze(0).expand(N,-1)

    def create_patch(self,color='black'):
        center = self.center.cpu().numpy()
        radius = self.radius
        circle = patches.Circle(center, radius, linewidth=3, edgecolor=color, facecolor='None')
        return circle
    
