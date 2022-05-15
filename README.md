## 1. Introduction
This project is based on PyTorch 1.11.0 and Python 3.9.7. It uses a GPU, of course it can run without a GPU (it requires a simple modification of the GPU part of the code). The hyperparameters are wrapped in utils and can be invoked directly in utils.

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