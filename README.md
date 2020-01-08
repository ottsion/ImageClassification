# Classification model based on small sample image data -- simple version


How to perform the task of image classification with ~~none~~ few samples in each category requires three steps:

1. Crawling data,
2. Training coarse-grained classification model
3. Training refined classification model

Data crawling using `beautifulsop` processing

Code data processing adopts` torchvision `processing

This model adopts the trained version of `vgg16` model of pytorch

## Crawling image data to be classified

Core code：`spyder/ImagesSpyder.py`

Fill the category information to be crawled into `Spyder/name.TXT`

After running, input the number of pictures to be crawled for each category

## Data handling

Select the crawled pictures and put them in the directory of `./data/train_old/`, remember to store them according to the principle of one directory for each category

Generally, it can be:

```
./data/train_old/0_person/
./data/train_old/1_flower/
...
```

Put the precious small amount of sample data you have into the directory `./data/train/`, which is the final classification target

## Model training

The core file is`recognition/models.py`

What needs to be revised:

1. Modify `n_classes` in the main function to the number of your categories
2. Model address `model_base_path`
3. Your GPU info : `os.environ[`CUDA_VISIBLE_DEVICES`]=`0,1``

As the topic said, you need to use different training sets to train the model twice. For the first time, you need to use the data in `train old`, and for the second time, you need to use the data in `train`

Remember to switch datasets when using. By default, you can use datasets in `train` for training. You can modify them freely

## Test Result

The following is the accuracy test results. Of course, we should not only focus on the accuracy, but also pay attention to the sample situation. It is better to use Gan to generate small samples to increase the sample size...

Base Model：

![](H:/code/ImageClassification/asserts/score_1.PNG)

   

Update Model：

![](H:/code/ImageClassification/asserts/score_2.PNG)

## Download&Contact
Basic model download:：[VGG16](https://pan.baidu.com/s/1LHTn89jgCr6MRCe2n4kFlw)

In case of failure, contact: `emsunfc@163.com`

