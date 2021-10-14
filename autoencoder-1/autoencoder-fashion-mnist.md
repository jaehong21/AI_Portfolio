---
description: fashion MNIST 데이터셋을 이용하여 기본적인 Autoencoder를 구현하는 페이지다.
---

# AutoEncoder fashion MNIST

```python
import torch
import torchvision
import torch.nn.functional as F
from torch import nn, optim
from torchvision import transforms, datasets

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
```

Autoencoder 구현에 앞서 필요한 라이브러리를  import 해준다.

```python
EPOCH = 10
BATCH_SIZE = 64
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print("Using Device: ", DEVICE)
```

```python
trainset =  datasets.FashionMNIST(
    root ='./.data/',
    train = True, download = True,
    transform = transforms.ToTensor()
)
train_loader = torch.utils.data.DataLoader(
    dataset = trainset,
    batch_size = BATCH_SIZE,
    shuffle = True,
    num_workers = 2
)
```

Fashion MNIST를 사용자의 폴더 내에 다운로드해준다. 

```python
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12 ,3)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid()
        )
        
        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return  encoded, decoded
```

```python
model = Autoencoder().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = nn.MSELoss()
```

Autoencoder에서 optimizer로 Adam, 그리고 손실값으로는 MSELoss를 사용했다.

```python
view_data = trainset.data[:5].view(-1, 28*28)
view_data = view_data.type(torch.FloatTensor)/255
```

```python
def train(model, train_loader):
    model.train()
    for idx, (x, label) in enumerate(train_loader):
        x = x.view(-1, 28*28).to(DEVICE)
        y = x.view(-1, 28*28).to(DEVICE)
        label = label.to(DEVICE)
        
        encoded, decoded = model(x)
        
        loss = criterion(decoded, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

```python
for epoch in range(1, EPOCH+1):
    train(model, train_loader)
    
     test_x = view_data.to(DEVICE)
     _, decoded_data = model(test_x)
     
        f, a = plt.subplot(2, 5, figsize=(5,2))
        print("[Epoch {}]".format(epoch))
        
        for i  in range(5):
           img = np.reshape(view_data.data.numpy()[i], (28, 28))
           a[0][i].imshow(img, cma p='gray')
              
        for i  in range(5):
           img = np.reshape(decoded_data.to("cpu").data.numpy()[i], (28, 28))
           a[0][i].imshow(img, cmap='gray')
           
        plt.show()
```

![Epoch 1\~3 and Epoch 8\~10](<../.gitbook/assets/image (21).png>)

