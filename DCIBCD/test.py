import torch
import os
from tensorboardX import SummaryWriter
from TrainUse import seed_torch, CDMetrics, CTime
from Loss import combine_loss

from nloaders import nloaders
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cuda")#cuda
LR = 2e-3
BATCH_SIZE = 30
from model import DCIBCD as trainNet
net = trainNet().to(device)
netName = "SRCNet_LEVIR-cd"
netName = "test/" + netName
writer = SummaryWriter(netName + "/")
criterion = combine_loss

data_path=''
test_loader = nloaders(BATCH_SIZE, dataset_dir=data_path)
for data in (test_loader):
    batch_img1, batch_img2, labels = data
    print("type1:",type(batch_img1),"type2", type(batch_img2),"labels", type(labels))


optimizer = torch.optim.AdamW(net.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8) # 学习率调整

def test():
    Ctime = CTime()
    if not os.path.exists(netName):
        os.mkdir(netName)
    Ctime.born()
    valMetrics = CDMetrics()
    length = len(test_loader)
    Ctime.start()
    net.load_state_dict(torch.load(f"sample//DSIFNCD//epoch_max.pth" ))
    net.eval()
    epoch = 0
    for data in test_loader: # 数据遍历
        with tqdm(total=length, desc=f"[epoch: {epoch + 1}/{length}]", ncols=125) as pbar:
            epoch = epoch + 1
            batch_img1, batch_img2, labels = data
            batch_img1 = batch_img1.float().to(device)
            batch_img2 = batch_img2.float().to(device)
            labels = labels.long().to(device)
            optimizer.zero_grad()
            cd_preds, cosDis, diff, sigma = net(batch_img1, batch_img2)
            cd_loss = criterion(cd_preds, labels, cosDis, diff, sigma)
            cd_preds = cd_preds[-1]
            _, cd_preds = torch.max(cd_preds, 1)

            valMetrics.set(cd_loss, cd_preds, labels, scheduler.get_last_lr())

            mean_val_metrics = valMetrics.get()

            pbar.set_postfix(
                {
                    "Loss": "{:.4f}/{:.4f}".format(
                        cd_loss.item(), mean_val_metrics["loss"]
                    ),
                    "F1": "{:.4f}".format(mean_val_metrics["f1"]),
                    "LR": "{:3.2e}".format(mean_val_metrics["lr"]),
                }
            )
            pbar.update(1)
    log_file_path = os.path.join(netName, "test_PC_DSIFNCD_0105_C.txt")
    with open(log_file_path, "a") as log_file:
        log_file.write(f"Test. [epoch_max298.pth]  ")
        for key, value in mean_val_metrics.items():
            log_file.write(f"{key}: {value}  ")
        log_file.write("\n")

    for k, v in mean_val_metrics.items():
        writer.add_scalars(str(k), {"test": v},  1)
    writer.close()
if __name__ == "__main__":
    n = 1 # 多次测试
    for i in range(n):
        test()