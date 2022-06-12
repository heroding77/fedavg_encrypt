## 1. Introduction
This project is based on PyTorch 1.11.0 and Python 3.9.7. It uses a GPU, and of course it can run without one (it requires a simple change to the GPU part of the code). The hyperparameter information is encapsulated in conf. Json of utils and can be invoked directly by modifying the JSON file.The data set currently adopts MNIST data set and CIFAR-10 data set, and the model is built to realize FedAvg based on gray and white pictures (single channel) and color pictures (three channels). The project code is simple and the notes are clear. If my project is helpful to you, please light up a small star, which will be the biggest encouragement and support for me!

## 2. How to use
### 2.1 pure
If you want pure FedAvg, Utils file set noise to 0, run the following code:
```bash
python server.py -c ./utils/conf.json
```
### 2.2 DP
If you want DP FedAvg, Utils file set noise to 1(laplace) or 2(gaussian), run the following code:
```bash
python server.py -c ./utils/conf.json
```

### 2.3 DP+paillier
If you want DP+paillier FedAvg, Utils file set noise to 1(laplace) or 2(gaussian), run the following code:
```bash
python server_encrypt.py -c ./utils/conf.json
```

## 3. Other documents

