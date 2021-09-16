# Pytorch tutorial 01: MNIST and Pretrained Model

###### tags: `Pytorch Tutorial`, `Deep Learning`

:::info
:bulb: **提示**:
1. 可先打開 colab 並將執行階段改為 GPU，搭配實作能更快速上手。
2. 預估時間: 1h 30m
:::

**教學目標:**

- [ ] 認識 Pytorch 的 tensor 物件
- [ ] 認識 Pytorch 的基本語法
- [ ] 利用 Pytorch 建立 CNN model
- [ ] 利用 Pytorch Pretrained model 進行 Transfer Learning


## Part 1: Basic Introduction

### Why Pytorch?

Pros:
1. 相較於 keras，Pytorch 有較大自訂彈性
2. 比 tensorflow 更容易閱讀與編寫
3. 比 tensorflow 容易安裝 (CUDA 與 cudnn 支援度比較高)
4. 比 tensorflow 穩定，不容易因版本問題而造成執行錯誤
5. 龐大的社群資源

Cons:
1. 比 keras 底層，需要更多學習時間
2. 發展比 tensorflow 晚

### Installation

本教學在 colab 上執行即可，不強制要求本地安裝 Pytorch。

#### CPU version (不支援 GPU 運算)

官網: [pytorch.org](https://pytorch.org/)

依照官網選擇你電腦的 os、安裝方式、版本...即可。

若要安裝CPU version， CUDA 選項請選擇 None。
![](https://i.imgur.com/tcw6ilY.png)

#### GPU version

請先確認GPU廠牌為 Nvidia，且有支援 CUDA。
需建立 Nvidia 帳號，還可以進行下載。

CUDA: [cuda-toolkit-archive](https://developer.nvidia.com/cuda-toolkit-archive)
cuDNN: [cuDNN Download](https://developer.nvidia.com/rdp/cudnn-archive)

:::info
:bulb: **提示**:
1. 須注意 Torch 1.7.1 版只支援 CUDA : 9.2, 10.1, 10.2, 11.0。
2. 依照 CUDA 版本來安裝對應的 cuDNN。
:::

只要選擇對應 CUDA 版本的 Pytorch 即可。

##### Check if CUDA version Pytorch installation succeed

若是有正確安裝 CUDA 版本的 Pytorch，則會回傳 True。
```python
#Boolean, True if GPU avalible.
torch.cuda.is_available()
```

#### Colab

不須特別設定或是安裝 Pytorch，可以直接使用。
```python
import torch
torch.__version__
```

### 免費GPU平台

#### Colab

![Colab](https://i.imgur.com/UTOmejr.png)

Pros:

1. 可與 google 雲端硬碟連結

Cons:

1. GPU 超過使用時間會斷線
2. GPU 是隨機分配

#### Kaggle

![Kaggle](https://i.imgur.com/k1VCXrK.png)

Pros:

1. GPU: Tesla P100 (每周30小時)
2. 沒超過用量不容易斷線

Cons:

1. 上傳資料較麻煩

## Part 2: Pytorch syntax

這邊僅簡單介紹基礎語法，更深入了解請參考: [DEEP LEARNING WITH PYTORCH: A 60 MINUTE BLITZ](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

### Tensor
Tensor 是在 Pytorch 中，用來進行計算的物件，類似 Pytorch 版本的 numpy array，但不完全一樣。

參考資料: [TORCH.TENSOR](https://pytorch.org/docs/stable/tensors.html)

```python
a = torch.tensor([[1,2,3]])
print('\na:\n', a)
print('\nType of a:\n', type(a))
print('\nShape of a:\n', a.shape)
print('-'*10)

b = torch.transpose(a,0,1)
print('\nb:\n', b)

c = torch.rand(2,3)
print('\nc:\n', c)

d = torch.zeros(4,2)
print('\nd:\n', d)
```
### Layers and Activation function
查看官方的說明文件可以快速了解用法。

**Pytorch document:**
- Layers:
    - [Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
    - [MaxPool2d](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d)
    - [Flatten](https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html#torch.nn.Flatten)
    - [Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear)
    - [BatchNorm2d (optional)](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html#torch.nn.BatchNorm2d)
    - [Dropout (optional)](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html#torch.nn.Dropout)
- Activation function:
    - [Sigmoid](https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html#torch.nn.Sigmoid)
    - [ReLU](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html#torch.nn.ReLU)
    - [Tanh](https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html#torch.nn.Tanh)
    
#### Convolution
##### 實作
```python=
import torch
import torch.nn as nn

input = torch.rand(1,1,3,3) # 輸入矩陣 (batch, channel, h, w)
print('輸入:\n',input)

conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2) # 實體化卷積層
print('kernel and bias:\n',conv_layer.state_dict())


output = conv_layer(input)
print('output:\n',output)
```

##### [optional] 實際例子
```python=
import torch
torch.manual_seed(0) # keep random seed
import urllib
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True) # 載入resnet
model.eval();

url, filename = ("https://wallpaperaccess.com/full/2773879.jpg", "cat.jpg") # download image

try:
  urllib.URLopener().retrieve(url, filename)
except:
  urllib.request.urlretrieve(url, filename)

input_image = Image.open(filename)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
print('原始圖片:')
plt.imshow(input_tensor.permute(1,2,0).cpu().detach().numpy())
plt.title('sample image')
plt.show()
```
```python=
import matplotlib.pylab as plt # plot module
import numpy as np

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

conv_layer = model.conv1 # 從resnet 的 conv1 提取kernel，共有64個kernel

conv_images = conv_layer(input_batch) # 把圖片丟入 conv_layer 中，得出卷積結果
kernel_images = conv_layer.state_dict()['weight'].cpu().detach().numpy() # 把 kernel 提取出來
print('\n[印出前 8 個 kernel 以及過完這個 kernel 的圖片]:\n')

fig=plt.figure(figsize=(10, 40))
columns = 2
rows = 8
for i in range(1, columns//2 * rows + 1):
    kernel = rgb2gray(kernel_images[i-1].transpose(1, 2, 0))
    img = conv_images[0][i-1].cpu().detach().numpy()
    fig.add_subplot(rows, columns, 2*i-1)
    plt.imshow(kernel, cmap='gray')
    fig.add_subplot(rows, columns, 2*i)
    plt.imshow(img, cmap='gray')

plt.show()
```
#### Pooling
##### 實作
<img src = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0110EN/notebook_images%20/chapter%206/6.1.3_max_pool_animation_2.gif" width = 500, align = "center"> 

```python=
image1=torch.zeros(1,1,4,4)
image1[0,0,0,:]=torch.tensor([1.0,2.0,3.0,-4.0])
image1[0,0,1,:]=torch.tensor([0.0,2.0,-3.0,0.0])
image1[0,0,2,:]=torch.tensor([0.0,2.0,3.0,1.0])
print('image1:\n',image1)


max1=torch.nn.MaxPool2d(2) # 實體化池化層
print('output:\n',max1(image1))
```
##### [optional] 實際例子
```python=
maxpool_layer = model.maxpool # 提取 resnet 中的 maxpool 層
maxpool_images = maxpool_layer(conv_images) # 將過完 convolution 後的圖片丟入 maxpool 層，得到結果
print('\n[印出前 8 個過完 convolution 以及再過完 max pooling 的圖片]:\n')

fig=plt.figure(figsize=(10, 40))
columns = 2
rows = 8
for i in range(1, columns//2 * rows + 1):
    img = conv_images[0][i-1].cpu().detach().numpy()
    maxpool_img = maxpool_images[0][i-1].cpu().detach().numpy()
    fig.add_subplot(rows, columns, 2*i-1)
    plt.imshow(img, cmap='gray')
    fig.add_subplot(rows, columns, 2*i)
    plt.imshow(maxpool_img, cmap='gray')

plt.show()
```
#### Flatten
```python=
image = torch.tensor([[[1,2,3],[4,5,6],[7,8,9]]])
print('image:\n',image)

m = nn.Flatten() # 實體化Flatten層
print(m(image))
```
#### Fully conntected layer(Dense)
```python=
input = torch.tensor([[1.,2.,3.,4.,5.]], requires_grad=True)
print('input:\n',input)

fc = nn.Linear(5,2) # 實體化全連接層
output = fc(input) # input 5維、output 2維
print('output:\n',output)
```

#### Activation layer
同樣的 input 經過不同的 activation layer 後，得到不同的結果。

```python=
input = torch.tensor([[-5,-1,0,3,8]]).float()
print('Input:',input)
print('-'*50)

relu = nn.ReLU() # 實體化ReLU層
out1 = relu(input)
print('ReLU:', out1)

tanh = nn.Tanh() # 實體化Tanh層
out2 = tanh(input)
print('Tanh:', out2)

sigmoid = nn.Sigmoid() # 實體化sigmoid層
out3 = sigmoid(input)
print('Sigmoid:', out3)

print('-'*50)
softmax = nn.Softmax(dim=1)  # 實體化Softmax層(一般用在最後一層輸出)
out4 = softmax(input)
print('Softmax:', out4)
```
## Part 3: MNIST
本節目標是使用 Pytorch 建立 LaNet-5 model，在 MNIST 資料集進行訓練與預測。

### 資料準備與前處理
#### 引入所需 module
```python=
import torch  
torch.manual_seed(0) # keep random seed
import torch.nn as nn 
import torchvision.transforms as transforms # pytorch image processing module
import torchvision.datasets as dsets # pytorch dataset
import matplotlib.pylab as plt 
import numpy as np  
from tqdm.notebook import tqdm
```

#### 多個前處理結合在變數 composed 中
1. resize()，將大小28x28的圖片轉成32*32
2. ToTensor()，把圖片轉成tensor(張量)才可以丟入model

參考資料: [transforms.Compose](https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.Compose)
```python=
IMAGE_SIZE = 32 # Original size: 28
composed = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))\ 
                            ,transforms.ToTensor(),])
```
#### 下載MNIST資料並將資料前處理
MNIST共包含了:
1. 6萬張的Training data
2. 1萬張的Testing data
```python=
train_dataset = dsets.MNIST(root='./data',\
                            train=True,\
                            download=True,\
                            transform=composed)    
                            
validation_dataset = dsets.MNIST(root='./data',\
                                train=False,\
                                download=True,\
                                transform=composed);
                                
print('Length of train_dataset:', len(train_dataset))
print('Length of validation_dataset:', len(validation_dataset))
```

#### 定義用來檢視 MNIST 資料的function
可利用這個 function 查看 MNIST 圖片。
```python=
def show_data(data_sample):
    plt.imshow(data_sample[0].numpy().reshape(IMAGE_SIZE, IMAGE_SIZE), cmap='gray')
    plt.title('y = '+ str(data_sample[1]))
```

查看圖片
```python=
num = 8 #<--- Any number you want 
show_data(train_dataset[num])
```

### Defining model
#### 選擇GPU進行訓練
```python=
if torch.cuda.is_available():
  device = torch.device('cuda:0')
  print('GPU is avalible.')
  print('Working on:', torch.cuda.get_device_name())
else:
  device = torch.device('cpu')
  print('GPU is not avalible.')
  print('Working on CPU')
```

#### [Try yourself] 定義模型
**請依照 LaNet-5 的架構圖，建立 model**
1. 需先求出 Kernel size。
2. Subsampling 使用 Average Pooling。
3. Stride 為 1。

![LaNet-5](https://miro.medium.com/max/4308/1*vUJ-XilD6_WECeQlOMThMQ.png)

Formula: 
$$
[\frac{(W−K+2P)}{S}]+1
$$
- W is the input volume - in this case 32
- K is the Kernel size - in this case 5
- P is the padding - in this case 0
- S is the stride - in this case 1

In LeNet-5, $[(32 - 5 + 0)/1] + 1 = 28$

```pythpn=
model = nn.Sequential(
        #===============START====================
        nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5),


        
        #===================END==================

        )

model.to(device) # moving to GPU
```

#### 檢視模型資訊
```python=
from torchsummary import summary
summary(model, (1, 32, 32))
```

#### 定義超參數
1. loss function
2. learning rate
3. optimizer
4. batch size
```python=
criterion = nn.CrossEntropyLoss()
learning_rate = 0.0005
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=1024)
```

#### 定義訓練函式
不同於 Keras 只需使用 model.fit()，Pytorch的訓練需要自己寫一個迴圈。

```python=
def train_model(model,train_loader,validation_loader,optimizer,n_epochs=4): 

  N_train=len(train_dataset) 
  N_test=len(validation_dataset) 
  
  train_acc=[] # store training accuracy
  cv_acc=[]   # store validation accuracy
  train_loss=[] # store training loss
  cv_loss=[]  #store validation loss

  for epoch in tqdm(range(n_epochs)):
    #==================training=========================
    training_loss = [] # 每個 mini-batch 的 training loss
    correct = 0 # 累積每個 mini-batch 答對的數量
    model.train() # switch model to train mode
    for x, y in train_loader:
      x, y = x.to(device), y.to(device) #把 data 移到GPU中
      optimizer.zero_grad() #初始化梯度(gradient)
      z = model(x) #把 x 丟入模型中得到預測值 z
      _, yhat = torch.max(z.data, 1)  #找到z中最大值的位置 yhat
      correct += (yhat == y).sum().item() #計算 yhat 跟 y 相同的數量
      loss = criterion(z, y) #比較 prediction z 與 label y ，計算loss值
      loss.backward() #計算反向傳播(back-propogation)的梯度值
      optimizer.step() #更新參數
      training_loss.append(loss.item()) #把每個 mini-batch 的loss值記錄下來
    train_loss.append(np.mean(training_loss)) #training loss 是每個 mini-batch的loss值取平均
    train_acc.append(correct / N_train) #accuracy 是把所有答對的數量 / 所有的data。 ACC = TP + FN / (TP+FP+TN+FN)

    #==================validation=========================
    training_loss = []
    correct = 0
    model.eval() # switch model to train mode
    with torch.no_grad(): 
      for x_test, y_test in validation_loader:
        x_test, y_test= x_test.to(device), y_test.to(device)
        z = model(x_test)
        _, yhat = torch.max(z.data, 1)
        correct += (yhat == y_test).sum().item()
        loss = criterion(z, y_test)
        training_loss.append(loss.item())
      cv_loss.append(np.mean(training_loss))
      cv_acc.append(correct / N_test)
     
  return  train_acc, cv_acc, train_loss, cv_loss
```
 
#### 訓練模型
```python=
train_acc, cv_acc, train_loss, cv_loss = train_model(model=model,n_epochs=4,train_loader=train_loader,validation_loader=validation_loader,optimizer=optimizer)
```

#### Learning Curve
訓練圖
```python=
plt.plot(train_acc,label='train_acc')
plt.plot(cv_acc ,label='cv_acc')
plt.title('train / valid  accuracy')
plt.xlabel('epochs')
plt.ylabel('acc')
#axes = plt.gca()
#axes.set_ylim([0.4, 1])
plt.legend()
plt.grid()
plt.show()

plt.plot(train_loss,label='train_cost')
plt.plot(cv_loss,label='cv_acc')
plt.title('train / valid  loss')
plt.xlabel('epochs')
plt.ylabel('loss')
#axes = plt.gca()
#axes.set_ylim([0, 1])
plt.legend()
plt.grid()
plt.show()
```
模型數據
```python=
print('[Training] ACC:',train_acc[-1])
print('[Training] LOSS:',train_loss[-1])
print('-'*10)
print('[Test] ACC:',cv_acc[-1])
print('[Test] LOSS:',cv_loss[-1])
```
## Part 4: 建立自己的CNN模型
試著自己拚出一個 model，須注意每一層的輸出維度!

**Formula:**
$$
[\frac{(W−K+2P)}{S}]+1
$$

- W is the input volume - in this case 32
- K is the Kernel size 
- P is the padding 
- S is the stride 


嘗試使用:
- nn.Conv2d()
- nn.AvgPool2d() / nn.MaxPool2d()
- nn.BatchNorm2d()
- nn.ReLU() / nn.Tanh() / nn.sigmoid()
- nn.Linear()
- nn.Flatten()
- 
更多資訊請看官方說明 <a href="https://pytorch.org/docs/stable/nn.html">torch.nn</a>

**參考範例:**
```
my_model = nn.Sequential(
        nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5),
        nn.BatchNorm2d(6),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Flatten(),
        nn.Dropout(p=0.5),
        nn.Linear(1176,10)
        )
```
### [Try yourself] 建立 model
```python=
my_model = nn.Sequential(
        #============================START================================
        nn.Conv2d(#modify here),

        #==============================END==============================
        )

my_model.to(device)
summary(my_model, (1, 32, 32))
```

### [Try yourself] 定義超參數
試著更換:
- learning_rate
- optimizer
- batch_size

<a href="https://pytorch.org/docs/stable/optim.html">pytorch: optimizer</a>

**參考範例:**
```
learning_rate = 0.0005
optimizer = torch.optim.Adam(my_model.parameters(), lr=learning_rate)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=1024)
```

實作
```python=
# modify yourself
#============================START================================
learning_rate = #modify here
optimizer = #modify here
train_loader torch.utils.data.DataLoader(dataset=train_dataset, batch_size= #modify here )
#==============================END==============================

criterion = nn.CrossEntropyLoss()
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=1024)

train_acc, cv_acc, train_loss, cv_loss = train_model(model=my_model,n_epochs=4,train_loader=train_loader,validation_loader=validation_loader,optimizer=optimizer)
```
Learning curve
```python=
plt.plot(train_acc,label='train_acc')
plt.plot(cv_acc ,label='cv_acc')
plt.title('train / valid  accuracy')
plt.xlabel('epochs')
plt.ylabel('acc')
#axes = plt.gca()
#axes.set_ylim([0.4, 1])
plt.legend()
plt.grid()
plt.show()

plt.plot(train_loss,label='train_cost')
plt.plot(cv_loss,label='cv_acc')
plt.title('train / valid  loss')
plt.xlabel('epochs')
plt.ylabel('loss')
#axes = plt.gca()
#axes.set_ylim([0, 1])
plt.legend()
plt.grid()
plt.show()

print('[Training] ACC:',train_acc[-1])
print('[Training] LOSS:',train_loss[-1])
print('-'*10)
print('[Test] ACC:',cv_acc[-1])
print('[Test] LOSS:',cv_loss[-1])
```
## Part 5: Pytorch Pretrained model:Transfer Learning
參考資料:
1. [torchvision.models](https://pytorch.org/docs/stable/torchvision/models.html)
2. [TRANSFER LEARNING FOR COMPUTER VISION TUTORIAL](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
```python=
import torchvision.models as models

model = models.resnext50_32x4d(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

model.to(device);
```

## [Optional] AlexNet
**AlexNet 架構:**
<hr>
<img src="https://miro.medium.com/max/3072/1*qyc21qM0oxWEuRaj-XJKcw.png" width="80%">

此部分在預處理時，需先將MNIST轉成RGB channel，並且對圖片做正規化處理，有興趣可以參考前處理的程式碼。

參考資料:
- [Pytorch|AlexNet](https://pytorch.org/hub/pytorch_vision_alexnet/)
- [原始論文](http://www.cs.toronto.edu/~hinton/absps/imagenet.pdf)

### 前處理
```python
import torch  # pytorch
torch.manual_seed(0) # keep random seed
import torch.nn as nn
import torchvision.transforms as transforms # image processing module
import torchvision.datasets as dsets # dataset
import matplotlib.pylab as plt # plot module
import numpy as np # matrix module
from tqdm.notebook import tqdm # 顯示進度條
```
```python
if torch.cuda.is_available():
  device = torch.device('cuda:0')
  print('GPU is avalible.')
  print('Working on:', torch.cuda.get_device_name())
else:
  device = torch.device('cpu')
  print('GPU is not avalible.')
  print('Working on CPU')
```

```python
IMAGE_SIZE = 224 # Original size: 28

composed = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                          transforms.ToTensor(),
                          transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```
```python
train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform=composed)
validation_dataset = dsets.MNIST(root='./data', train=False, download=True, transform=composed)

print('Length of train_dataset:', len(train_dataset))
print('Length of validation_dataset:', len(validation_dataset))
```

```python
def show_data(data_sample):
    image = data_sample[0].numpy().reshape(3, IMAGE_SIZE, IMAGE_SIZE)
    image = np.transpose(image, axes=[1, 2, 0])
    plt.imshow(image)
    plt.title('y = '+ str(data_sample[1]))
```

檢視MNIST圖片
```python
num = 8 #<--- Any number you want 
show_data(train_dataset[num])
```
### [Try yourself] 定義模型
#### [Try yourself] 定義model
```python
class AlexNet(nn.Module):
    # Contructor
    def __init__(self):
        super(AlexNet, self).__init__()
        self.feature = nn.Sequential(
        #==================START========================


        #===================END=========================
        )
    # Prediction
    def forward(self, x):
        x = self.feature(x)
        return x

model = AlexNet()

model.to(device) # moving to GPU
```
#### 檢視模型
```python=
from torchsummary import summary
summary(model, (3, 224, 224))
```
#### 訓練函數
```python
def train_model(model,train_loader,validation_loader,optimizer,n_epochs=4):
  N_test=len(validation_dataset)
  N_train=len(train_dataset)
  train_acc=[] 
  cv_acc=[]   
  train_loss=[] 
  cv_loss=[]   

  for epoch in tqdm(range(n_epochs)):
    #training=============================================
    training_loss = []
    correct = 0
    model.train() # switch model to train mode
    for x, y in train_loader:
      x, y = x.to(device), y.to(device)
      optimizer.zero_grad()
      z = model(x)
      _, yhat = torch.max(z.data, 1)
      correct += (yhat == y).sum().item()
      loss = criterion(z, y)
      loss.backward()
      optimizer.step()
      training_loss.append(loss.item()) 
    train_loss.append(np.mean(training_loss))
    train_acc.append(correct / N_train)

    #validation===========================================
    training_loss = []
    correct = 0
    model.eval() # switch model to train mode
    with torch.no_grad(): 
      for x_test, y_test in validation_loader:
        x_test, y_test= x_test.to(device), y_test.to(device)
        z = model(x_test)
        _, yhat = torch.max(z.data, 1)
        correct += (yhat == y_test).sum().item()
        loss = criterion(z, y_test)
        training_loss.append(loss.item())
      cv_loss.append(np.mean(training_loss))
      cv_acc.append(correct / N_test)
     
  return  train_acc, cv_acc, train_loss, cv_loss
```
#### 超參數
```python
criterion = nn.CrossEntropyLoss()
learning_rate = 0.0005
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=256)
```

#### 訓練
```python
train_acc, cv_acc, train_loss, cv_loss = train_model(model=model,n_epochs=4,train_loader=train_loader,validation_loader=validation_loader,optimizer=optimizer)
```

### Learning curve
```python
plt.plot(train_acc,label='train_acc')
plt.plot(cv_acc ,label='cv_acc')
plt.title('train / valid  accuracy')
plt.xlabel('epochs')
plt.ylabel('acc')
#axes = plt.gca()
#axes.set_ylim([0.4, 1])
plt.legend()
plt.grid()
plt.show()

plt.plot(train_loss,label='train_cost')
plt.plot(cv_loss,label='cv_acc')
plt.title('train / valid  loss')
plt.xlabel('epochs')
plt.ylabel('loss')
#axes = plt.gca()
#axes.set_ylim([0, 1])
plt.legend()
plt.grid()
plt.show()
```
```python
print('[Training] ACC:',train_acc[-1])
print('[Training] LOSS:',train_loss[-1])
print('-'*10)
print('[Test] ACC:',cv_acc[-1])
print('[Test] LOSS:',cv_loss[-1])
```

## 參考答案
### LaNet-5
Method 1
```python
model = nn.Sequential(
          nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5),
          nn.AvgPool2d(kernel_size=2),
          nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5),
          nn.AvgPool2d(kernel_size=2),
          nn.Flatten(),
 
          nn.Linear(5*5*16,120),
          nn.Tanh(),
 
          nn.Linear(120,84),
          nn.Tanh(),
 
          nn.Linear(84,10),
        )
model.to(device) # moving to GPU
```

Method 2
```python
class CNN(nn.Module):
    # Contructor
    def __init__(self, out_1=6, out_2=16):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=out_1, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=out_1, out_channels=out_2, kernel_size=5)
        self.avgpool = nn.AvgPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(5*5*16, 120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
 
    # Prediction
    def forward(self, x):
        x = self.avgpool(self.conv1(x))
        x = self.avgpool(self.conv2(x))
        x = self.flatten(x)
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.fc3(x)
 
        return x
 
model = CNN(out_1=6, out_2=16)
model.to(device) # moving to GPU
```

### AlexNet
```python
class AlexNet(nn.Module):
    # Contructor
    def __init__(self,out):
        super(AlexNet, self).__init__()
        self.out = out
        self.feature = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(6, 6))
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=9216, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=self.out, bias=True)
        )
    # Prediction
    def forward(self, x):
        x = self.feature(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
 
model = AlexNet(10)
 
model.to(device) # moving to GPU
```