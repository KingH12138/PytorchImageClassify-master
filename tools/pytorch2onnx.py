import torch
import torchvision

dummy_input = torch.randn(1, 3, 224, 224, device='cuda')
input_names = ["input_1"]
output_names = ["output_1"]
model = torch.load(r"F:\PycharmProjects\mmaction2-0.22.0\weights\slowfast_temporal_max_focal_alpha3_gamma1_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb-345618cd.pth")
torch.onnx.export(model, dummy_input, "SlowFast.onnx", verbose=True, input_names=input_names, output_names=output_names)
