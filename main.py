import torch

model = torch.load(r'D:\PythonCode\Pytorch-ImageClassification-master\workdir\exp_7_28_8_1_pytorch-imageclassification-master\checkpoints\best_f1.pth')
layers = list(model.named_children())[0:3]
print()