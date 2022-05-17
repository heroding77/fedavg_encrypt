## 1. 简介
本项目基于PyTorch 1.11.0和Python 3.9.7。它使用GPU，当然它可以在没有GPU的情况下运行(它需要对代码的GPU部分进行简单的修改)。超参数信息封装在utils的conf.json中，可以修改json文件直接调用。

## 2. 使用方法
如果你只是想使用FedAvg，修改conf.json中的noise设置为0, 运行如下代码：
```bash
python server.py -c ./utils/conf.json
```
### 2.2 差分隐私
如果你想使用基于差分隐私的FedAvg，修改conf.json中的noise设置为1（拉普拉斯机制）或者2（高斯机制），sigma用来调节噪声幅度，运行如下代码：
```bash
python server.py -c ./utils/conf.json
```

### 2.3 DP+paillier
如果你想使用基于差分隐私和同态加密的FedAvg，修改conf.json中的noise设置为1（拉普拉斯机制）或者2（高斯机制），sigma用来调节噪声幅度，运行如下代码：
```bash
python server_encrypt.py -c ./utils/conf.json
```
## 3. 其他文件
new_paillier.py文件中的加密算法是用来加解密文本信息，test.py是用来测试Paillier的用法，可以忽略。