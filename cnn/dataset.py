from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor




train_dataset = MNIST(
    root='./data',
    train = True,
    download = True,
    transform = ToTensor()
    )



train_loader = DataLoader(
     dataset=train_dataset,
     batch_size=16,
     shuffle=True,
     #num_workers=2
 )



# for epoch in range(10):
#      for batch_x, batch_y in dataloader:
#         pass
