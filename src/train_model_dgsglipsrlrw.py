import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.datasets import DatasetFolder
from torch import distributed as dist
import torchvision
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt
import time
import skvideo.io, skvideo.utils
from skimage.color import rgb2lab
from sklearn import metrics
import random
from pytorchvideo.transforms import RandAugment
import pickle
import sys

torch.autograd.set_detect_anomaly(True)

#Parameter
lr = 0.00001
batch_size = 1
device = "cuda" if torch.cuda.is_available() else "cpu"
error = nn.CrossEntropyLoss()

#To change

end_dataset = sys.argv[1] # e.g. "dgs"
tl_path = sys.argv[2] # e.g. "lrw_dgs"
finetune = int(sys.argv[3]) #0 or 1 for finetune
load_weight = sys.argv[4] # e.g. "lrw_dgs"
replace_fc = bool(int(sys.argv[5])) #0 or 1 to reinitialize last layer

pthSave = "./Weights/{}.pth".format(tl_path)
tensorboardSave = "./measures/{}".format(tl_path)
trainPath = "./{}/train".format(end_dataset)
valPath = "./{}/val".format(end_dataset)
testPath = "./{}/test".format(end_dataset)
patience = 200
loadWeights = bool(finetune)
weightPath = "./Weights/{}.pth".format(load_weight)

if os.path.isfile(pthSave):
    raise Exception("Weights exist already")
    

max_frames =30


klassen = {0: "52",1: "94",2: "137",3: "242",4: "273",5: "326",
6: "329",7: "339",8: "354",9: "364", 10: "369", 11: "425"}

klasse = ["52", "94", "137", "242", "273", "326", "329","339","354", "364", "369", "425"]


class Netz(nn.Module):
    def __init__(self):
        super(Netz, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.BatchNorm3d(16),

            nn.Conv3d(16, 16, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Dropout3d(0.5, inplace=False),
            nn.BatchNorm3d(16),

            nn.Conv3d(16, 32, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Dropout3d(0.5, inplace=False),
            nn.BatchNorm3d(32),

        )
        self.fc_0 = nn.Sequential(
            nn.Linear(2*256*max_frames, 15)
        )
        self.fc_1 = nn.Sequential(
            nn.Linear(2*256*max_frames, 15)
        )

        self.fc_2 = nn.Sequential(
            nn.Linear(2*256*max_frames, 15)
        )

        #1728
        self.lstm = nn.GRU(1152, 256, 1, bidirectional=True)
        #self.lstm = nn.GRU(27648, 256, 1, bidirectional=True)


        self.lstm2 = nn.GRU(512, 256, 1, bidirectional=True)

        self.batch = nn.BatchNorm3d(3)
        self.dropout1 = nn.Dropout(0.5, inplace=False)
        self.dropout2 = nn.Dropout(0.5, inplace=False)

    def forward(self, x, task_id : int):
        x = self.batch(x)
        # B C T H W
        x = self.conv(x)
        # T B C H W
        x = x.permute(2, 0, 1, 3, 4).contiguous()
        # T B C*H*W
        x = x.view(x.size(0), x.size(1), -1)

        self.lstm.flatten_parameters()
        self.lstm2.flatten_parameters()

        # T B Features
        x, _ = self.lstm(x)
        x = self.dropout1(x)

        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x = x.permute(1,0,2).contiguous()
	# B T Features
        x = x.view(x.size(0), -1)

        if task_id == 0:
            x = self.fc_0(x)
        elif task_id == 1:
            x = self.fc_1(x)
        elif task_id == 2:
            x = self.fc_2(x)
        else:
            assert False, 'Bad Task ID passed'

        return x


def init_process_group():
    """
    Join the process group and return whether this is the rank 0 process,
    the CUDA device to use, and the total number of GPUs used for training.
    """
    rank = int(os.getenv('RANK', 0))
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    num_gpus = int(os.getenv('WORLD_SIZE', 1))
    dist.init_process_group('nccl')
    return rank == 0, torch.device(f'cuda:{local_rank}'), num_gpus

randaug = RandAugment(4,3)

def loaderAug(path):
    # Video einlesen
    data = torchvision.io.read_video(path)[0]

    # Dimension: Time, Channels, Height, Width
    data = data.permute(0, 3, 1, 2)

    # Augmentierung
    data = randaug(data)

    # Normalisieren von 0-255 in 0-1
    data = data / 255.0

     # C T H W
    return data.permute(1, 0, 2, 3)


def loader(path):
    # Video einlesen
    data = torchvision.io.read_video(path)[0]

    # Dimension: Time, Channels, Height, Width
    data = data.permute(0, 3, 1, 2)

    # Normalisieren von 0-255 in 0-1
    data = data / 255.0

    # C T H W
    return data.permute(1, 0, 2, 3)



train = DatasetFolder(trainPath, loader=loaderAug, extensions=("mp4"))
val = DatasetFolder(valPath, loader=loader, extensions=("mp4"))
test = DatasetFolder(testPath, loader=loader, extensions=("mp4"))
#klasse = val.classes
#klasssen = dict()
#print(klasse)
#print()
#print(klassen)

train_dgs = DatasetFolder("./dgs2/train", loader=loaderAug, extensions=("mp4"))
val_dgs = DatasetFolder("./dgs2/val", loader=loader, extensions=("mp4"))
test_dgs = DatasetFolder("./dgs2/test", loader=loader, extensions=("mp4"))

train_glipsr = DatasetFolder("./fg_glips_r2/train", loader=loaderAug, extensions=("mp4"))
val_glipsr = DatasetFolder("./fg_glips_r2/val", loader=loader, extensions=("mp4"))
test_glipsr = DatasetFolder("./fg_glips_r2/test", loader=loader, extensions=("mp4"))

train_glipsm = DatasetFolder("./fg_lrw2/train", loader=loaderAug, extensions=("mp4"))
val_glipsm = DatasetFolder("./fg_lrw2/val", loader=loader, extensions=("mp4"))
test_glipsm = DatasetFolder("./fg_lrw2/test", loader=loader, extensions=("mp4"))



#for b in val.class_to_idx:
#    klassen[val.class_to_idx[b]] = b

if __name__ == "__main__":
    print(pthSave)
    is_rank0, device, num_gpus = init_process_group()
    torch.cuda.set_device(device)
    
    train_sampler_dgs = torch.utils.data.distributed.DistributedSampler(train_dgs, num_replicas=num_gpus, rank=int(os.getenv('RANK', 0)))
    val_sampler_dgs = torch.utils.data.distributed.DistributedSampler(val_dgs, num_replicas=num_gpus, rank=int(os.getenv('RANK', 0)))
    test_sampler_dgs = torch.utils.data.distributed.DistributedSampler(test_dgs, num_replicas=num_gpus, rank=int(os.getenv('RANK', 0)))
    
    trainloader_dgs = DataLoader(train_dgs, shuffle=False, batch_size=64, num_workers=4, sampler=train_sampler_dgs)
    valloader_dgs = DataLoader(val_dgs, shuffle=False, batch_size=64, num_workers=4, sampler=val_sampler_dgs)
    testloader_dgs = DataLoader(test_dgs, shuffle=False, batch_size=64, num_workers=4, sampler=test_sampler_dgs)

    train_sampler_glipsm = torch.utils.data.distributed.DistributedSampler(train_glipsm, num_replicas=num_gpus, rank=int(os.getenv('RANK', 0)))
    val_sampler_glipsm = torch.utils.data.distributed.DistributedSampler(val_glipsm, num_replicas=num_gpus, rank=int(os.getenv('RANK', 0)))
    test_sampler_glipsm = torch.utils.data.distributed.DistributedSampler(test_glipsm, num_replicas=num_gpus, rank=int(os.getenv('RANK', 0)))
    
    trainloader_glipsm = DataLoader(train_glipsm, shuffle=False, batch_size=64, num_workers=4, sampler=train_sampler_glipsm)
    valloader_glipsm = DataLoader(val_glipsm, shuffle=False, batch_size=64, num_workers=4, sampler=val_sampler_glipsm)
    testloader_glipsm = DataLoader(test_glipsm, shuffle=False, batch_size=64, num_workers=4, sampler=test_sampler_glipsm)

    train_sampler_glipsr = torch.utils.data.distributed.DistributedSampler(train_glipsr, num_replicas=num_gpus, rank=int(os.getenv('RANK', 0)))
    val_sampler_glipsr = torch.utils.data.distributed.DistributedSampler(val_glipsr, num_replicas=num_gpus, rank=int(os.getenv('RANK', 0)))
    test_sampler_glipsr = torch.utils.data.distributed.DistributedSampler(test_glipsr, num_replicas=num_gpus, rank=int(os.getenv('RANK', 0)))
    
    trainloader_glipsr = DataLoader(train_glipsr, shuffle=False, batch_size=64, num_workers=4, sampler=train_sampler_glipsr)
    valloader_glipsr = DataLoader(val_glipsr, shuffle=False, batch_size=64, num_workers=4, sampler=val_sampler_glipsr)
    testloader_glipsr = DataLoader(test_glipsr, shuffle=False, batch_size=64, num_workers=4, sampler=test_sampler_glipsr)
    
    Model = Netz()

    #checkpoint = torch.load("./ckpt_hps_high.pt")

    Model.to(device)
    #Model = nn.parallel.DistributedDataParallel(Model,find_unused_parameters=True)

    #Model.load_state_dict(checkpoint['model_state_dict'])

    """
    if loadWeights:
        Model.load_state_dict(torch.load(weightPath))
        Model = Model.module
        if replace_fc:
            Model.fc[-1] = nn.Linear(500, 15)
    """
    Model.to(device)
    #Model = nn.parallel.DistributedDataParallel(Model)         
    checkpoint = torch.load("./ckpt_hps_dgs-glipsr-lrw.pt")
    Model.load_state_dict(checkpoint['model_state_dict'])    

    optimizer = torch.optim.Adam(Model.parameters(), lr=lr)

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    #---Test bei ungesehenen Bildern
    def test(dataloader, task_id):
        with torch.no_grad():
            Model.eval()
            Model.to(device)
            correct = 0
            total = 0

            actual = []
            pred = []
            classes= []

            for images, labels in dataloader:
                images, labels = Variable(images).to(device, dtype=torch.float), Variable(labels).to(device)
                output = Model.forward(images, task_id)
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
               
                #for topk accuracy batchsize must be 1!
                #_, topk_indices = torch.topk(output, 2)
                #print(labels, topk_indices)
                #correct += int(labels[0] in topk_indices)
                
                #print(output.shape, labels.shape, predicted.shape)

                #actual.append(klassen[predicted.item()])
                #if labels.item() not in classes:
                #    classes.append(klassen[labels.item()])
                #pred.append(klassen[labels.item()])

        #cf = confusion_matrix(pred, actual,normalize="true")
        
        #print(metrics.classification_report(actual, pred, digits=4))
        #fig, ax = plt.subplots(figsize=(10,10))
        #sns.heatmap(cf, annot=True, fmt='.2f', xticklabels=klasse, yticklabels=klasse)
        #plt.ylabel('Actual')
        #plt.xlabel('Predicted')        
        #plt.savefig("confusionbody_04.png")
        Model.train()
        return 100 * (correct / total)

    # ---Training
    def train(model):
        # Optimierer & Loss
        error_glipsm = nn.CrossEntropyLoss()
        error_dgs = nn.CrossEntropyLoss()

        model.train()  # trainierbar
        besteVal = 0  # beste Genauigkeit beim Validierungsset
        epochs_no_improve = 0  # Anzahl Epochen ohne Verbesserung
        best_epochs_no_improve = 0
        epoch = 0
        writer = SummaryWriter(tensorboardSave)

        epoch = checkpoint['epoch'] + 1

        # Epochen
        for i in range(epoch, 10000):
            correct, correct_dgs, correct_glipsm = 0, 0, 0
            total , total_dgs, total_glipsm = 0, 0, 0

            zipped = zip(trainloader_glipsm, trainloader_glipsr, trainloader_dgs)
            for j, ((glipsm_batch_X, glipsm_batch_y), (glipsr_batch_X, glipsr_batch_y), (dgs_batch_X, dgs_batch_y)) in enumerate(zipped):
                # Variables: Tensoren mit zur Berechnnug des Gradienten
                glipsm_batch_X = Variable(glipsm_batch_X).to(device, dtype=torch.float)
                glipsm_batch_y = Variable(glipsm_batch_y).to(device)
                glipsr_batch_X = Variable(glipsr_batch_X).to(device, dtype=torch.float)
                glipsr_batch_y = Variable(glipsr_batch_y).to(device)
                dgs_batch_X = Variable(dgs_batch_X).to(device, dtype=torch.float)
                dgs_batch_y = Variable(dgs_batch_y).to(device)

                # Gradienten-Werte zuruecksetzen
                optimizer.zero_grad()
                # prediction
                output_dgs = model.forward(dgs_batch_X, task_id = 1)
                loss_dgs = error(output_dgs, dgs_batch_y)

                output_glipsm = model.forward(glipsm_batch_X, task_id = 0)
                loss_glipsm = error(output_glipsm, glipsm_batch_y)

                output_glipsr = model.forward(glipsr_batch_X, task_id = 2)
                loss_glipsr = error(output_glipsr, glipsr_batch_y)

                loss = loss_dgs + loss_glipsm + loss_glipsr
                loss.backward()

                optimizer.step()
                # Genauigkeit Ausgabe
                _, predicted_dgs = torch.max(output_dgs.data, 1)
                total_dgs += dgs_batch_y.size(0)
                correct_dgs += (predicted_dgs == dgs_batch_y).sum().item()

                _, predicted_glipsm = torch.max(output_glipsm.data, 1)
                total_glipsm += glipsm_batch_y.size(0)
                correct_glipsm += (predicted_glipsm == glipsm_batch_y).sum().item()

                total = total_dgs 
                correct = correct_dgs 
  

            aktuelleVal_glipsm = test(valloader_glipsm, 0)
            aktuelleVal_dgs = test(valloader_dgs, 1)

            print("Epoche", i, str(total), "/", len(trainloader_dgs) * 2 * batch_size, "Loss: ", loss_dgs.data.item(), " ",
                  "Genauigkeit: ", str(correct / total*100), "Val-Genauigkeit_dgs: ", str(aktuelleVal_dgs))
            writer.add_scalars("Genauigkeit", {"Train": np.float(str(correct / total*100)), "Val_dgs": np.float(aktuelleVal_dgs)}, i)
            writer.flush()

            if i % 200 == 0:
                torch.save({
            'epoch': i,
            'model_state_dict': Model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, "./ckpt2_hps_dgs-glipsr-lrw.pt")

            # Gewichtungen mit bester Genauigkeit beim Validierungsset speichern
            if aktuelleVal_dgs > besteVal:
                besteVal = aktuelleVal_dgs
                epochs_no_improve = 0
                torch.save(Model.state_dict(), pthSave)
            else:
                epochs_no_improve += 1
                if aktuelleVal_dgs > 95 and epochs_no_improve >= patience:
                    break


        print("Bestes Ergebnis:", besteVal)
        writer.close()

    # Training
    train(Model)

    print(test(testloader_dgs))

