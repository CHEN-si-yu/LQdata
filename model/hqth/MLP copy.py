
import torch
import torch.nn as nn
import torch.nn.functional as F
from kan import KAN
class MLP(nn.Module):
    def __init__(self, factor_num):
        super(MLP, self).__init__()
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
            KAN([128,32]),
            nn.LayerNorm(32),
            nn.GELU(),
        )
        # 输出层
        self.output_layer = KAN([32,1])
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

    def forward(self, tsdata):
        x = tsdata.float() 
        x = self.input_layter(x)
        x = self.net(x)
        x = self.output_layer(x)
        return x  

if __name__ == "__main__":
    model = MLP(1400)
    # print(model)
    x = torch.randn(32, 1400)
    print(f"x.shape: {x.shape}")  # 应该是 [32, 1400]
    output = model(x)
    print(f"output.shape: {output.shape}")  # 应该是 [32, 1]