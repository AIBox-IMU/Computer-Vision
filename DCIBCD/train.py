import torch
import os
from tensorboardX import SummaryWriter
from TrainUse import seed_torch, CDMetrics, CTime
from Loss import combine_loss

from nloaders import nloaders
import warnings

warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cuda")#cuda

LR = 2e-3
BATCH_SIZE = 15
EPOCH = 300
PRI_EPOCH = 0

from model import DCIBCD as trainNet

net = trainNet().to(device)
netName = "DSIFNCD"

netName = "DCIBCD_DSIFNCD/" + netName

writer = SummaryWriter(netName + "/")

seed_torch(seed=3407)

criterion = combine_loss
dataset_path = 'datasets//DSIFNCD//'
train_loader, val_loader = nloaders(BATCH_SIZE, dataset_dir=dataset_path)
print("数据加载完成")
optimizer = torch.optim.AdamW(net.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8) # 学习率调整

if __name__ == "__main__":
    Ctime = CTime()
    if not os.path.exists(netName):
        os.mkdir(netName)
    if PRI_EPOCH > 0:
        net.load_state_dict(torch.load(netName + "/epoch_" + str(PRI_EPOCH) + ".pth"))
    if PRI_EPOCH != 0:
        for epoch in range(0, PRI_EPOCH):
            scheduler.step()
    maxf1 = 0
    maxepoch = 0
    Ctime.born()
    for epoch in range(PRI_EPOCH, EPOCH):
        trainMetrics = CDMetrics()
        valMetrics = CDMetrics()
        net.train()
        length = len(train_loader)
        import time
        from tqdm import tqdm
        train_times = []
        val_times = []

        Ctime.start()
        with tqdm(total=length, desc=f"[epoch: {epoch + 1}/{EPOCH}]", ncols=125) as pbar:
            train_start_time = time.time()
            for data in train_loader:
                batch_img1, batch_img2, labels = data
                batch_img1 = batch_img1.float().to(device)
                batch_img2 = batch_img2.float().to(device)
                labels = labels.long().to(device)
                optimizer.zero_grad()
                forward_start = time.time()
                cd_preds, cosDis, diff, sigma = net(batch_img1, batch_img2)
                forward_time = time.time() - forward_start
                loss_start = time.time()
                cd_loss = criterion(cd_preds, labels, cosDis, diff, sigma)
                loss_time = time.time() - loss_start
                loss = cd_loss
                backward_start = time.time()
                loss.backward()
                optimizer.step()
                backward_time = time.time() - backward_start
                cd_preds = cd_preds[-1]
                _, cd_preds = torch.max(cd_preds, 1)
                trainMetrics.set(cd_loss, cd_preds, labels, scheduler.get_last_lr())
                mean_train_metrics = trainMetrics.get()
                pbar.set_postfix({
                    "Loss": f"{cd_loss.item():.4f}/{mean_train_metrics['loss']:.4f}",
                    "F1": f"{mean_train_metrics['f1']:.4f}",
                    "LR": f"{mean_train_metrics['lr']:3.2e}",
                    "FwdTime": f"{forward_time:.3f}s",
                    "BwdTime": f"{backward_time:.3f}s"
                })
                pbar.update(1)
                del batch_img1, batch_img2, labels
            train_epoch_time = time.time() - train_start_time
            train_times.append(train_epoch_time)
            log_file_path = os.path.join(netName, "train_log.txt")
            with open(log_file_path, "a") as log_file:
                log_file.write(f"Training. [{epoch + 1}]  ")
                for key, value in mean_train_metrics.items():
                    log_file.write(f"{key}: {value}  ")
                log_file.write(f"TrainTime: {train_epoch_time:.2f}s\n")
            val_start_time = time.time()
            net.eval()
            with torch.no_grad():
                length = len(val_loader)
                for i, data in enumerate(val_loader, 0):
                    batch_img1, batch_img2, labels = data
                    batch_img1 = batch_img1.float().to(device)
                    batch_img2 = batch_img2.float().to(device)
                    labels = labels.long().to(device)
                    val_forward_start = time.time()
                    cd_preds, cosDis, diff, sigma = net(batch_img1, batch_img2)
                    val_forward_time = time.time() - val_forward_start
                    cd_loss = criterion(cd_preds, labels, cosDis, diff, sigma)
                    cd_preds = cd_preds[-1]
                    _, cd_preds = torch.max(cd_preds, 1)
                    valMetrics.set(cd_loss, cd_preds, labels, scheduler.get_last_lr())
                    mean_val_metrics = valMetrics.get()
                    del batch_img1, batch_img2, labels
                    print(f"  Test: {i}/{length} [FwdTime: {val_forward_time:.3f}s]\r", end="")
            val_epoch_time = time.time() - val_start_time
            val_times.append(val_epoch_time)
            maxepoch = epoch + 1
            torch.save(net.state_dict(), netName + "/epoch_" + str(maxepoch) + ".pth")
            if mean_val_metrics["f1"] > maxf1:
                maxf1 = mean_val_metrics["f1"]
                if epoch + 1 - maxepoch <= 5 and maxepoch != 0:
                    os.remove(netName + "/epoch_" + str(maxepoch) + ".pth")
                maxepoch = epoch + 1
                torch.save(net.state_dict(), netName + "/epoch_max" + str(maxepoch) + ".pth")
            print(f"\n  Loss: {mean_val_metrics['loss']:.05f} | F1: {mean_val_metrics['f1']:.04f} | "
                  f"MAX-F1: {maxf1:.04f} ({maxepoch})")
            print(f"  TrainTime: {train_epoch_time:.2f}s | ValTime: {val_epoch_time:.2f}s | "
                  f"AvgTrainTime: {sum(train_times) / len(train_times):.2f}s | "
                  f"AvgValTime: {sum(val_times) / len(val_times):.2f}s")
            log_file_path = os.path.join(netName, "test_log.txt")
            with open(log_file_path, "a") as log_file:
                log_file.write(f"Test. [{epoch + 1}]  ")
                for key, value in mean_val_metrics.items():
                    log_file.write(f"{key}: {value}  ")
                log_file.write(f"ValTime: {val_epoch_time:.2f}s\n")
            for k, v in mean_val_metrics.items():
                writer.add_scalars(str(k), {"test": v}, epoch + 1)
            writer.add_scalar("Time/Train", train_epoch_time, epoch + 1)
            writer.add_scalar("Time/Validation", val_epoch_time, epoch + 1)
        Ctime.end()
        Ctime.show(epoch, PRI_EPOCH, EPOCH)
    writer.close()