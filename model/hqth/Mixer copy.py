
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model import *
from kan import KAN
class Mixer(nn.Module):
    def __init__(self, factor_num, time_steps=1, market=128):
        super(Mixer, self).__init__()


        input_dim = factor_num

        # 输入层
        self.input_layter = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.2)
        )

        # 隐含层
        self.net = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # 混合层
        channels = 128
        scale_dim = 1
        self.mixer = MultTime2dMixer(time_steps, channels, scale_dim=scale_dim)
        self.conv = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=1, stride=1)
        self.dropout_conv = nn.Dropout(p=0.2)        # Conv 后
        self.dropout_channel = nn.Dropout(p=0.3)     # channel_fc 后

        # 输出层
        self.output_layer = KAN([128 * 3,1])
        
        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)


    def forward(self, inputs):

        inputs = inputs.float()
        inputs = self.input_layter(inputs)
        inputs = self.net(inputs)

        inputs = inputs.unsqueeze(1)
        x = inputs.permute(0, 2, 1)
        x = self.conv(x)
        x = self.dropout_conv(x)
        x = x.permute(0, 2, 1)

        y = self.mixer(inputs, x)
        y = y.view(y.shape[0], -1)
        y = self.dropout_channel(y)

        y = self.output_layer(y)
        return y

if __name__ == "__main__":
    model = Mixer(1400)
    # print(model)
    x = torch.randn(32, 1400)
    print(f"x.shape: {x.shape}")  # 应该是 [32, 1400]
    output = model(x)
    print(f"output.shape: {output.shape}")  # 应该是 [32, 1]