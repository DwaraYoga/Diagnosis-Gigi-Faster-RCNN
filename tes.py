import torch
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

checkpoint = torch.load('best_model.pth', map_location=DEVICE)
print(checkpoint['data']['NC'])