# 关键词：
## 解析解（analytical solution）:

当模型和损失函数形式较为简单时，上面的误差最小化问题的解可以直接用公式表达出来。这类解叫作解析解（analytical solution）

## 数值解（numerical solution）：

然而，大多数深度学习模型并没有解析解，只能通过优化算法有限次迭代模型参数来尽可能降低损失函数的值。这类解叫作数值解（numerical solution）。

## 导数和偏导数：

导数与偏导数本质是一致的，都是当自变量的变化量趋于0时，函数值的变化量与自变量变化量比值的极限。直观地说，偏导数也就是函数在某一点上沿坐标轴正方向的的变化率。区别在于： 　

导数，指的是一元函数中，函数y=f(x)在某一点处沿x轴正方向的变化率； 　

偏导数，指的是多元函数中，函数y=f(x1,x2,…,xn)在某一点处沿某一坐标轴（x1,x2,…,xn）正方向的变化率。

## 损失函数：
由于我创建的模型只能尽量拟合时间的结果，但是模型结果和真实结果是有出入的，这部分差别我们可以把它称之为**【损失】**。

假设真实结果是是y，创建的模型为

http://chart.googleapis.com/chart?cht=tx&chl=\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a})


## 梯度
梯度的提出只为回答一个问题： 　

函数在变量空间的某一点处，沿着哪一个方向有最大的变化率？ 　

梯度定义如下： 　

函数在某一点的梯度是这样一个向量，它的方向与取得最大方向导数的方向一致，而它的模为方向导数的最大值。 　

这里注意三点： 　

1）梯度是一个向量，即有方向有大小； 

2）梯度的方向是最大方向导数的方向； 　

3）梯度的值是最大方向导数的值。

## 梯度下降算法
可参考：梯度下降算法https://blog.csdn.net/red_stone1/article/details/80212814
如果函数f(θ)是凸函数，那么就可以使用梯度下降算法进行优化。梯度下降算法的公式我们已经很熟悉了 θ=θ0−η⋅∇f(θ0) 其中，θ0是自变量参数，即下山位置坐标，η是学习因子，即下山每次前进的一小步（步进长度），θ是更新后的θ0，即下山移动一小步之后的位置。
矢量计算

    向量相加的一种方法是，将这两个向量按元素逐一做标量加法。
    向量相加的另一种方法是，将这两个向量直接做矢量加法。 两者相比，矢量计算的计算效率更高。

## torch.rand和torch.randn有什么区别？

y = torch.rand(5,3) ：均匀分布 

y = torch.randn(5,3) ：标准正态分布

### 均匀分布

torch.rand(*sizes, out=None) → Tensor 返回一个张量，包含了从区间[0, 1)的均匀分布中抽取的一组随机数。张量的形状由参数sizes定义。 参数:

sizes (int...) - 整数序列，定义了输出张量的形状
out (Tensor, optinal) - 结果张

一个均匀分布，一个是标准正态分布。 

例子：
```
torch.rand(2, 3)
0.0836 0.6151 0.6958
0.6998 0.2560 0.0139
[torch.FloatTensor of size 2x3]
```

### 标准正态分布

torch.randn(*sizes, out=None) → Tensor 返回一个张量，包含了从标准正态分布（均值为0，方差为1，即高斯白噪声）中抽取的一组随机数。张量的形状由参数sizes定义。 参数: sizes (int...) - 整数序列，定义了输出张量的形状 out (Tensor, optinal) - 结果张量 

例子：
```
torch.randn(2, 3)
0.5419 0.1594 -0.0413
-2.7937 0.9534 0.4561
[torch.FloatTensor of size 2x3]
```
torch.tensor()

torch.tensor()仅仅是Python的函数，函数原型是：

torch.tensor(data, dtype=None, device=None, requires_grad=False) 其中data可以是：list, tuple, array, scalar等类型。 torch.tensor()可以从data中的数据部分做拷贝（而不是直接引用），根据原始数据类型生成相应的torch.LongTensor，torch.FloatTensor，torch.DoubleTensor。
```
>>> a = torch.tensor([1, 2])
>>> a.type()
'torch.LongTensor'
```
```
>>> a = torch.tensor([1., 2.])
>>> a.type()
'torch.FloatTensor'
```
```
>>> a = np.zeros(2, dtype=np.float64)
>>> a = torch.tensor(a)
>>> a.type()
torch.DoubleTensor
```

![Image](image/)
