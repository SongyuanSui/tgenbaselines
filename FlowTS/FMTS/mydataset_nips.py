import os
import sys
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

print("PROJECT_ROOT =", PROJECT_ROOT)
print("sys.path[0] =", sys.path[0])

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error

from engine.solver import Trainer
from Utils.io_utils import load_yaml_config, instantiate_from_config


# =========================
# 1. 修改成你的数据路径
# =========================
DATA_ROOT = '../../datasets/synthetic_u'   # 改成你的目录
TRAIN_PATH = os.path.join(DATA_ROOT, 'train_ts.npy')
VALID_PATH = os.path.join(DATA_ROOT, 'valid_ts.npy')
TEST_PATH  = os.path.join(DATA_ROOT, 'test_ts.npy')


# =========================
# 2. 读取数据
# shape: (N, L, D)
# =========================
train = np.load(TRAIN_PATH).astype(np.float32)
valid = np.load(VALID_PATH).astype(np.float32)
test  = np.load(TEST_PATH).astype(np.float32)

print("train shape:", train.shape)
print("valid shape:", valid.shape)
print("test  shape:", test.shape)

assert train.ndim == 3 and valid.ndim == 3 and test.ndim == 3
assert train.shape[1] == valid.shape[1] == test.shape[1]
assert train.shape[2] == valid.shape[2] == test.shape[2]


# =========================
# 3. 数据集类
# =========================
class MyDataset(Dataset):
    def __init__(self, data, regular=True, pred_length=24):
        super().__init__()
        self.samples = data
        self.sample_num = data.shape[0]
        self.regular = regular

        self.mask = np.ones_like(data, dtype=bool)
        self.mask[:, -pred_length:, :] = 0

    def __getitem__(self, ind):
        x = self.samples[ind, :, :]
        if self.regular:
            return torch.from_numpy(x).float()
        mask = self.mask[ind, :, :]
        return torch.from_numpy(x).float(), torch.from_numpy(mask)

    def __len__(self):
        return self.sample_num


# =========================
# 4. 训练集 dataloader
# =========================
train_dataset = MyDataset(train, regular=True)

dataloader = DataLoader(
    train_dataset,
    batch_size=64,     # 可调
    shuffle=True,
    num_workers=8,
    drop_last=False,
    pin_memory=True,
    sampler=None
)


# =========================
# 5. 载入配置
# =========================
class Args_Example:
    def __init__(self) -> None:
        self.config_path = './Config/mydataset.yaml'   # 你自己的 FlowTS config
        self.gpu = 0

args = Args_Example()
configs = load_yaml_config(args.config_path)

device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

model = instantiate_from_config(configs['model']).to(device)
trainer = Trainer(config=configs, args=args, model=model, dataloader={'dataloader': dataloader})


# =========================
# 6. 训练
# =========================
trainer.train()
# trainer.load(10)   # 如果已经训好就注释 train，改用 load


# =========================
# 7. forecasting 测试
# =========================
_, seq_length, feat_num = test.shape
pred_length = 24   # 可改成 12 / 24 / 48 等

test_dataset = MyDataset(test, regular=False, pred_length=pred_length)

real = test.copy()

test_dataloader = DataLoader(
    test_dataset,
    batch_size=test.shape[0],   # 一次全测；如果显存不够可以改小
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    sampler=None
)

# conditional sampling / restore
sample, *_ = trainer.restore(test_dataloader, shape=[seq_length, feat_num])

mask = test_dataset.mask
mse = mean_squared_error(sample[~mask], real[~mask])

print(f"Forecasting MSE: {mse:.6f}")

# 保存 log
log_str_pre = 'mydataset_forecasting ' + ' '.join(
    f"{k}={v}" for k, v in os.environ.items() if 'hucfg' in k
)
with open('log.txt', 'a') as f:
    f.write(log_str_pre + f" mse={mse}\n")


# =========================
# 8. 可视化
# 单变量数据只画 1 条
# =========================
import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 12

num_plot = min(2, test.shape[0])   # 画前两个样本
for idx in range(num_plot):
    plt.figure(figsize=(15, 3))
    plt.plot(
        range(0, seq_length - pred_length),
        real[idx, :(seq_length - pred_length), 0],
        color='c',
        linestyle='solid',
        label='History'
    )
    plt.plot(
        range(seq_length - pred_length - 1, seq_length),
        real[idx, -pred_length - 1:, 0],
        color='g',
        linestyle='solid',
        label='Ground Truth'
    )
    plt.plot(
        range(seq_length - pred_length - 1, seq_length),
        sample[idx, -pred_length - 1:, 0],
        color='r',
        linestyle='solid',
        label='Prediction'
    )
    plt.tick_params('both', labelsize=15)
    plt.subplots_adjust(bottom=0.1, left=0.05, right=0.99, top=0.95)
    plt.legend()
    plt.savefig(f'./mydataset_forecast_{idx}.png')
    plt.close()