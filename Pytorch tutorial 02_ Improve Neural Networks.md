# Pytorch tutorial 02: Improve Neural Networks
###### tags: `Pytorch Tutorial`, `Deep Learning`
**教程目標**
- [ ] 認識 Bias / Variance
- [ ] 認識並應用 Regularization / Dropout
- [ ] 認識並應用 BatchNorm / Learning rate decay
- [ ] 認識 Early stopping

**NN 示意圖**
![](https://i.imgur.com/3x8K1ZG.png)

## Bias / Variance
透過看 Training loss，可得知模型的 bias 趨勢。
透過看 Cross Validation (Dev) loss，可得知模型 variance 趨勢。

![](https://i.imgur.com/6A0KjKK.png)
![](https://i.imgur.com/5oWxKZ9.png)
![](https://i.imgur.com/6rzTxYw.png)

## Basic recipe for ML
![](https://i.imgur.com/kOzMKMU.png)

high bias? $\rightarrow$ Bigger network / Train longer / NN architecture $\rightarrow$ high bias? / Hight variance?

Hight variance?  $\rightarrow$ More data / Regularization / NN architecture $\rightarrow$ high bias? / Hight variance?

## Regularization
在 loss function 上加上 Regularization term，使得更新權重w時，會將權重w變小(又稱weight decay)，dicision boundary 會變得比較平滑，避免over-fitting現象。

![](https://i.imgur.com/vT5n9Dp.png)

### intuition
假設lambda很大，在backpropogation更新參數時要降低cost function時，會把權重w降低，導致每個權重都接近0，在nn中，若權重接近0的node就像是把node拿掉，導致整個nn只剩下bias連接著，剩下的nn就近似互相連接的logistic regression，沒辦法做太複雜的決定。
![](https://i.imgur.com/7PbDJN9.png)

假設激活函數使用tanh，而Regularizatio導致w變小，間接導致z變小(z^l = w^l * a^l-1 + b^l)，導致tanh(z)在很線性的區間，若整個nn的layers都在很接近線性的區間，則可以避免over-fitting。

![](https://i.imgur.com/qdkmDz1.png)

### Regularization in Pytorch
在 Pytorch 的 Regularization (L2 regularlization)，是以 parameter 的形式存在於 optimizer 中。
[Adam](https://pytorch.org/docs/stable/optim.html#torch.optim.Adam)
![](https://i.imgur.com/ACMfDM2.png)
```
torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
```
## Dropout Regularization
在每個layer中，設定Dropout機率，使得每個layer中的node有一定機率被移除，導致nn變成比較簡單的網路，再利用backpropogation去調整剩下的node，藉此避免over-fitting。

如果擔心某層有較大的over-fitting機率，可以將該層的drop out的kept prop調小，若是較簡單的layer，則kept prop調大，也可以在input layer設定drop out(一般kept prop = 0.9)。
![](https://i.imgur.com/xQFCjQ6.png)

### Dropout in Pytorch
[Dropout Document](https://i.imgur.com/uSA0JRg.png)
```
nn.Dropout(p=0.1)
```

## BatchNorm
在 nn 內部的 feature scaliing，把每個 mini-batch 的 feature 調整成 mean = 0, variance = 1，能夠加快收斂速度。

batch_norm 有一些些 regulaeiztion 的效果，越小的 batch_size 的 regulaeiztion 小果越大(越多雜訊)。

### intuition
![](https://i.imgur.com/ELh4HAz.png)
![](https://i.imgur.com/N5bwQWO.png)
![](https://i.imgur.com/8dfAufz.png)
![](https://i.imgur.com/S16Wp9p.png)
![](https://i.imgur.com/7TR99p3.png)

使得在 loss surface 能夠更快收斂。
![](https://i.imgur.com/sWNRHeW.png)


### BatchNorm in Pytorch
[BatchNorm document](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html#torch.nn.BatchNorm2d)
```
nn.BatchNorm2d(num_features=16)
```

## Learning rate decay
固定的 learning rate，可能導致在訓練後期一直無法收斂。
因此會希望 learning rate 在後期能縮小。

有需多調整 Learning rate的方法，請參考 [How to adjust learning rate](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
[ReduceLROnPlateau](https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau)
```
>>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
>>> scheduler = ReduceLROnPlateau(optimizer, 'min')
>>> for epoch in range(10):
>>>     train(...)
>>>     val_loss = validate(...)
>>>     # Note that step should be called after validate()
>>>     scheduler.step(val_loss)
```

## Early stopping
讓訓練提早停止，避免發生 over-fitting。

![](https://i.imgur.com/pwTyqg9.png)

目前 pytorch 並無提供 early-stopping 的 function，所以需要自己寫。

psudo code:
```python=
min_loss = None
count = 0

if min_loss == None:
    min_loss = val_loss
    save(model)
    
elif val_loss < min_loss:
    count += 1

else:
    min_loss = val_loss
    count = 0
    save(model)
```

[參考 github ](https://github.com/Bjarten/early-stopping-pytorch)

## Hypterparameters
- Tier 1
    - Learning rate
- Tier 2
    - momentum
    - hidden units
    - mini-batch size
- Tier 3
    - learning rate decay
    - layers
    - Adma($\beta1$, $\beta2$, $\epsilon$)