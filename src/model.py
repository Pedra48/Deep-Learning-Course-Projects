import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.encoder = nn.Sequential(
            nn.Linear(
                in_features= 1,
                out_features=self.args.n_hidden,
            ),
            nn.ReLU()
        )
        self.Processor = nn.ModuleList(
            [
            nn.Sequential(
                nn.Linear(in_features=self.args.n_hidden,
                      out_features=self.args.n_hidden,),
                nn.ReLU()
            )
            for _ in range(self.args.n_layers)
        ])
        
        self.decoder = nn.Sequential(
            nn.Linear(
                in_features=self.args.n_hidden,
                out_features= 1
            )
        )
        
        
    def forward(self,x):
        x = self.encoder(x)
        for module in self.Processor:
            x = module(x)
        x = self.decoder(x)
        return x
        