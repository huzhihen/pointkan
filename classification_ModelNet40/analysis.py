import torch
import fvcore.nn
import fvcore.common
from fvcore.nn import FlopCountAnalysis
from models import pointKAN

model = pointKAN()
model.cuda()  # 确保模型在GPU上
model.eval()

inputs = torch.randn((1, 3, 1024)).cuda()  # 确保输入数据在GPU上
k = 1024.0

# 计算FLOPs
try:
    flops = FlopCountAnalysis(model, inputs).total()
    print(f"Flops : {flops}")
    flops = flops / (k ** 3)
    print(f"Flops : {flops:.1f}G")
except Exception as e:
    print("Error in Flops calculation:", e)

# 计算参数量
try:
    params = fvcore.nn.parameter_count(model)[""]
    print(f"Params : {params}")
    params = params / (k ** 2)
    print(f"Params : {params:.1f}M")
except Exception as e:
    print("Error in parameter count:", e)
