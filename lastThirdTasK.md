# 什么是凸函数
凹凸函数本质是描述函数斜率增减的。

语义上凸为正，代表斜率在增加（单调不减）。凹为负，代表斜率在减少。

通常在实际中，最小化的函数有几个极值，所以最优化算法得出的极值不确实是否为全局的极值。

对于一些特殊的函数，凸函数与凹函数，任何局部极值也是全局极致，因此如果目标函数是凸的或凹的，那么优化算法就能保证是全局的。

定义1：集合Rc⊂En是凸集，如果对每对点x1,x2⊂Rc，每个实数α,0<α<1，点
x=αx1+(1−α)x2
位于Rc，即x∈Rc。

效果上，如果任何两点x1,x2∈Rc用直线相连，x1,x2之间线上的每个点都在Rc中，那么Rc是凸的。如果存在点不在Rc中，那么该集合是非凸的，凸集合如图1所示。

凸的概念也可以用到函数上。

定义2：
我们称定义在凸集Rc上的函数f(x)为凸的，如果对每对x1,x2∈Rc与每个实数α,0<α<1，不等式
f[αx1+(1−α)x2]≤αf(x1)+(1−α)f(x2)
满足。如果x1≠x2
f[αx1+(1−α)x2]<αf(x1)+(1−α)f(x2)
满足，那么f(x)是严格凸的。

如果φ(x)定义在凸集Rc上且f(x)=−φ(x)是凸的，那么φ(x)是凹的。如果f(x)是严格凸的，那么φ(x)是严格凹的。
上述定义中的不等式，左边是点x1,x2之间某处的f(x)值，而右边是基于线性插值得到的f(x)的近似，因此如果任何两点的线性插值大于函数的值，那么该函数就是凸的，图2a，b中的函数为凸的，2c为非凸的。



# 生成对抗网路
在机器学习领域判别模型是一种对未知数据y 与已知数据 x 之间关系进行建模的方法。判别模型是一种基于概率理论的方法。已知输入变量x ，判别模型通过构建条件概率分布 P(y|x)预测 y 。

与生成模型不同，判别模型不考虑 x 与 y 间的联合分布。对于诸如分类和回归问题，由于不考虑联合概率分布，采用判别模型可以取得更好的效果。而生成模型在刻画复杂学习任务中的依赖关系方面则较判别模型更加灵活。大部分判别模型本身是监督学习模型，不易扩展用于非监督学习过程。实践中，需根据应用的具体特性来选取判别模型或生成模型。

贯穿全书，我们已经讨论了如何做预测。我们使用深度神经网络来学习数据点到标签的映射。这种学习我们称之为判别模型，比如我们想要判别出猫和狗的照片。分类和回归属于判别学习。而神经网络通过反向传播来进行训练，从而颠覆了我对判别模型在巨大而复杂的数据集上表现的认识。就在这5～6年见，对于高分辨率的图像的分类精度等级，已经从无效识别提升到与人类识别相同的等级了。










