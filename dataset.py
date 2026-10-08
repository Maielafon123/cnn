from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor




train_dataset = MNIST(
    root='./data',
    train = True,
    download = True,
    transform = ToTensor()
    )



dataloader = DataLoader(
     dataset=train_dataset,
     batch_size=16,
     shuffle=True,
     #num_workers=2
 )

batch_x, batch_y = next(iter(dataloader))


print(batch_x.shape)
print(batch_y.shape)

print(batch_x.dtype)
print(batch_y.dtype)
print(batch_y)

# for epoch in range(10):
#      for batch_x, batch_y in dataloader:
#         pass
