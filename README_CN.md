# 小样本图片数据下的分类模型--简陋版

如何在~~一无所有~~每类别很少样本的情况下进行图片分类任务，你需要三步骤：
1. 爬取数据， 
2. 训练粗粒度分类模型
3. 训练精细化分类模型

数据爬取使用`BeautifulSoup`处理

代码数据处理采用`torchvision`处理

本模型采用已训练好的pytorch版本的`VGG16`模型

## 爬取需要分类的图片数据

核心代码：`spyder/ImagesSpyder.py`

将需要爬去的类别信息填入`spyder/name.txt`中

运行后输入每个类别需要爬取的图片数量即可

## 数据搬运

将爬取的图片进行挑选后放入`./data/train_old/`目录下，记得按照每个类别一个目录的原则进行存放

一般可以为：

```
./data/train_old/0_person/
./data/train_old/1_flower/
...
```

将你拥有的珍贵的少量样本数据放入`./data/train/`目录下，这才是最终的分类目标

## 模型训练

主体文件为`recognition/models.py`
需要修改的地方：
1. main函数中的`n_classes`，修改为你的类别个数
2. 模型地址，`model_base_path`
3. 你的GPU信息`os.environ['CUDA_VISIBLE_DEVICES']='0,1'`

正如题目所说，你需要使用不同的训练集训练两遍本模型，第一次使用`train_old`中数据
第二次使用`train`中数据

使用时记得切换数据集，默认使用`train`中数据集训练，你可以自由修改

基础模型下载：[VGG16](https://pan.baidu.com/s/1LHTn89jgCr6MRCe2n4kFlw)
如失效可联系`emsunfc@163.com`

