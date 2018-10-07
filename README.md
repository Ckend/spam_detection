# 基于CNN的垃圾信息自动识别

各目录及文件作用：

    data: 存放训练数据及文本处理后的数据
    runs: 存放训练结果
    data_helpers.py: 处理训练数据
    text_cnn.py: 定义CNN类
    train.py: 训练模型
    eval.py: 测试模型


## 使用

1. 进入 train.py 并适宜调整参数进行训练，结果会被保存在run中
2. 进入 eval.py ，适当调整测试信息(如checkpoint_dir、x_raw等)，进行模型测试

## 参考

    1.Implementing a CNN for Text Classification in TensorFlow:
    http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow

    2.【NLP】TensorFlow实现CNN用于中文文本分类
    http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow