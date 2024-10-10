import torch
import torch.nn as nn
from climax.cnn_blocks import  ResidualBlock,PeriodicConv2D

class ResNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        history=1,
        hidden_channels=128,
        activation="leaky",
        norm: bool = True,
        dropout: float = 0.1,
        n_blocks: int = 50,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels * history
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        elif activation == "leaky":
            self.activation = nn.LeakyReLU(0.3)
        else:
            raise NotImplementedError(f"Activation {activation} not implemented")

        self.image_proj = PeriodicConv2D(
            self.in_channels, 2*hidden_channels, kernel_size=7, padding=3, stride=2
        )
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    2*hidden_channels,
                    2*hidden_channels,
                    activation=activation,
                    norm=True,
                    dropout=dropout,
                )
                for _ in range(n_blocks)
            ]
        )

        if norm:
            self.norm = nn.BatchNorm2d(2*hidden_channels)
        else:
            self.norm = nn.Identity()
#        self.final = PeriodicConv2D(
#            hidden_channels, out_channels, kernel_size=7, padding=3
#        )
        self.final = nn.ConvTranspose2d(2*hidden_channels, out_channels, kernel_size=7, stride=2, padding=3, output_padding=1)
    def forward(self, x):
        if len(x.shape) == 5:  # x.shape = [B,T,C,H,W]
            x = x.flatten(1, 2)
        # x.shape = [B,T*C,H,W]
        x = self.image_proj(x)
        for block in self.blocks:
            x = block(x)
        yhat = self.final(self.activation(self.norm(x)))
        yhat = yhat[:,:,:121,:256]
        # yhat.shape = [B,C,H,W]
        return yhat
def test():
    model = ResidualBlock(in_channels=3, out_channels=128)
    x = torch.randn(1, 3, 32, 64)
    y = model(x)
    print(y.shape)

def main():
    model = ResNet(in_channels=69, out_channels=69,n_blocks=28)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(n_parameters)
    x = torch.randn(1, 69, 32, 64)
    y = model(x)
    print(y.shape)
if __name__ == '__main__':
    main()
