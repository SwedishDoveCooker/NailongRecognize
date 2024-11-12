# 一个奶龙鉴别模型
理论上这个库可以进行所有二分类模型训练，只要自己改掉train和test以及相应negative文件夹内的图片再运行train.py就可以做自己想要的图片分类（检测图片中是否有某元素，输出是或否）

支持gif和视频，原理遍历所有帧，有一张有就输出有，否则输出无

fork自项目 https://github.com/spawner1145/NailongRecognize 原作者spawner1145

使用方法(训练模型):

```
pip install numpy torch torchvision scikit-learn imbalanced-learn opencv-python
python train.py
```

模型会保存到 `nailong.pth` 下

bot:

```
nb create
```

输入项目名及依赖器等, 如QQ使用`Onebot`

将 `plugins` 下代码放入插件目录中, 并在 `pyproject.toml` 中声明插件

```
nb run
```

