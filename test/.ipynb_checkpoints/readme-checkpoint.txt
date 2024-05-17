下游任务测试代码
数据集：在/gdownstream文件夹下，是六万交易的那个版本，二十万的那个还没跑
数据集的缓存文件：/data.pickle
训练方式：batch_size = 1 ,三分类
启动方式：python main.py
文件夹下应有：
apex文件夹：库
graph：早期编程留下的屎山的一部分，基本没有用处
downstream：数据集
esperberto-merges，esperberto-vocab：tokenizer的配置
gnn_module:gnn模块
data_preprocessor:代码预处理
myembedding：embedding模块
training_pipeline:训练模块
transformer_module:transformer模块
main：模型启动
nn_embedding_model.pth，gnn_model.pth，transformer_model.pth：预训练模型。
output文件夹：存放训练完以后的模型

当前环境如下：（与预训练同）
absl-py                    2.1.0
accelerate                 0.28.0
aiohttp                    3.9.1
aiosignal                  1.3.1
antlr4-python3-runtime     4.8
appdirs                    1.4.4
async-timeout              4.0.3
attrs                      23.2.0
axial-positional-embedding 0.2.1
base58                     2.1.1
bitarray                   2.9.2
cachetools                 5.3.2
certifi                    2023.11.17
charset-normalizer         3.3.2
ckzg                       1.0.0
click                      8.1.7
colorama                   0.4.6
contourpy                  1.2.0
cycler                     0.12.1
Cython                     3.0.6
cytoolz                    0.12.3
dgl                        2.0.0+cu121
docker-pycreds             0.4.0
docstring-parser           0.15
einops                     0.7.0
eth-abi                    2.2.0
eth-account                0.5.9
eth-hash                   0.7.0
eth-keyfile                0.5.1
eth-keys                   0.3.4
eth-rlp                    0.2.1
eth-typing                 2.3.0
eth-utils                  1.9.5
filelock                   3.13.1
fonttools                  4.47.2
frozenlist                 1.4.1
fsspec                     2023.12.2
future                     0.18.3
gitdb                      4.0.11
GitPython                  3.1.42
google-auth                2.27.0
google-auth-oauthlib       1.2.0
googledrivedownloader      0.4
grpcio                     1.60.0
hexbytes                   0.3.1
huggingface-hub            0.21.4
hydra-core                 1.0.0
idna                       3.6
importlib-metadata         7.0.1
importlib-resources        6.1.1
ipfshttpclient             0.8.0a2
isodate                    0.6.1
Jinja2                     3.1.3
joblib                     1.3.2
jsonargparse               4.27.3
jsonschema                 3.2.0
jsonschema-specifications  2023.12.1
kiwisolver                 1.4.5
lightning-utilities        0.10.1
littleutils                0.2.2
lmdb                       1.4.1
local-attention            1.9.0
lru-dict                   1.2.0
lxml                       5.1.0
Markdown                   3.5.2
markdown-it-py             3.0.0
MarkupSafe                 2.1.4
matplotlib                 3.8.2
mdurl                      0.1.2
more-itertools             10.2.0
mpmath                     1.3.0
multiaddr                  0.0.9
multidict                  6.0.4
netaddr                    1.2.1
networkx                   3.2.1
numpy                      1.26.3
nvidia-cublas-cu12         12.1.3.1
nvidia-cuda-cupti-cu12     12.1.105
nvidia-cuda-nvrtc-cu12     12.1.105
nvidia-cuda-runtime-cu12   12.1.105
nvidia-cudnn-cu12          8.9.2.26
nvidia-cufft-cu12          11.0.2.54
nvidia-curand-cu12         10.3.2.106
nvidia-cusolver-cu12       11.4.5.107
nvidia-cusparse-cu12       12.1.0.106
nvidia-nccl-cu12           2.18.1
nvidia-nvjitlink-cu12      12.3.101
nvidia-nvtx-cu12           12.1.105
oauthlib                   3.2.2
ogb                        1.3.6
omegaconf                  2.1.2
outdated                   0.2.2
packaging                  23.2
pandas                     2.2.0
parsimonious               0.8.1
performer-pytorch          1.1.4
pillow                     10.2.0
pip                        23.3.1
portalocker                2.8.2
protobuf                   3.20.1
psutil                     5.9.8
ptvsd                      4.3.2
pyasn1                     0.5.1
pyasn1-modules             0.3.0
pycryptodome               3.20.0
pyDeprecate                0.3.0
pyg-lib                    0.3.1+pt21cu121
Pygments                   2.17.2
pyparsing                  3.1.1
pyrsistent                 0.20.0
python-dateutil            2.8.2
python-louvain             0.16
pytorch-lightning          2.1.0
pytz                       2023.3.post1
pyunormalize               15.1.0
PyYAML                     5.4.1
rdflib                     7.0.0
rdkit-pypi                 2021.9.3
referencing                0.34.0
regex                      2023.12.25
requests                   2.31.0
requests-oauthlib          1.3.1
rich                       13.7.0
rlp                        2.0.1
rpds-py                    0.18.0
rsa                        4.9
sacrebleu                  2.4.0
safetensors                0.4.2
scikit-learn               1.4.0
scipy                      1.12.0
sentry-sdk                 1.43.0
setproctitle               1.3.3
setuptools                 69.0.3
six                        1.16.0
smmap                      5.0.1
sympy                      1.12
tabulate                   0.9.0
tensorboard                2.15.1
tensorboard-data-server    0.7.2
tensorboardX               2.4.1
threadpoolctl              3.2.0
tokenizers                 0.19.1
toolz                      0.12.1
torch                      2.1.0+cu121
torch-cluster              1.6.3+pt21cu121
torch-geometric            1.7.2
torch-scatter              2.1.2+pt21cu121
torch-sparse               0.6.18+pt21cu121
torch-spline-conv          1.2.2+pt21cu121
torchaudio                 2.1.0
torchdata                  0.7.1
torchmetrics               0.7.0
tqdm                       4.66.1
transformers               4.40.2
triton                     2.1.0
typeshed-client            2.4.0
typing_extensions          4.9.0
tzdata                     2023.4
urllib3                    2.1.0
varint                     1.0.2
wandb                      0.16.6
web3                       5.25.0
websockets                 9.1
Werkzeug                   3.0.1
wheel                      0.41.2
yarl                       1.9.4
zipp                       3.17.0