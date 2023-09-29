import sys, os

from torch.functional import split
import torch, torchvision, cv2
from torchvision import transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from vaes_models import *
from vaes_losses import *
import numpy as np
from torch.utils.data import Dataset, DataLoader
import copy

torch.manual_seed(111)
np.random.seed(111) 
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

def data_transforms():

    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)

    transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.CenterCrop(148),
                                    transforms.Resize(64),
                                    transforms.ToTensor(),
                                    SetRange])
    return transform

def show_image(x):
    x = (x + 1) / 2
    fig = plt.figure()
    plt.imshow(np.transpose(x.cpu().numpy(), (1, 2, 0)))
    plt.show()

# celeba_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Resize(64)]))
# celeba_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Resize(64)]))

# celeba_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=data_transforms())
# celeba_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=data_transforms())

celeba_trainset = datasets.CelebA(root='./data', split='train', download=True, transform=data_transforms())
celeba_testset = datasets.CelebA(root='./data', split='test', download=True, transform=data_transforms())

train_dataloader = torch.utils.data.DataLoader(celeba_trainset, batch_size=64, shuffle=True, drop_last=True)
val_dataloader = torch.utils.data.DataLoader(celeba_testset, batch_size=32, shuffle=False, drop_last=True)
dataiter = iter(train_dataloader)
images, labels = dataiter.next()
# show_image(torchvision.utils.make_grid(images))

print("Training dataset size: ", len(celeba_trainset))
print("Validation dataset size: ", len(celeba_testset))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

save_model = True
latent_dim = 128 # 128
epochs = 100
exp_name = "dont_careness_storage2/VAE_Celeba_KLD_N_0_4_no_batchtrack" # "models/WAE_MMD" # 
rv = True

if rv:
    model_name = exp_name + '_RV.pt'
else:
    model_name = exp_name + '.pt'

# torch.autograd.set_detect_anomaly(True)
model = VanillaVAE(in_channels=images[0].shape[0], latent_dim=latent_dim, rv=rv).to(device)
# model = BetaTCVAE(in_channels=images[0].shape[0], latent_dim=latent_dim, rv=rv).to(device)
# model.load_state_dict(torch.load(model_name))
loss_function = VanillaLoss
# loss_function = BetaTCVAELoss
# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# print('Network\'s trainable parameters:', count_parameters(model))

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
print("Start training VAE...")
best_val_loss = 999999
train_loss = []
kld_loss = []
val_loss = []
criterion = torch.nn.MSELoss()
for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    total_kld = 0
    total_val_loss = 0
    for batch_idx, (x, _) in enumerate(train_dataloader):
        x = x.to(device)

        optimizer.zero_grad()
        with torch.autograd.profiler.profile(use_cuda=True,with_stack=True) as prof:
            x_hat, mean, log_var = model(x)
        print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=10))

        loss, kld = loss_function(x, x_hat, mean, log_var, celeba_trainset, rv)

        # x_hat, mean, log_var, z = model(x)

        # loss, _, kld = loss_function(x_hat, x, mean, log_var, z, celeba_trainset, model, rv)
        # print(loss.item(), kld.item())
        total_train_loss += loss.item()
        total_kld += kld.item()

        loss.backward()
        optimizer.step()
    total_train_loss = total_train_loss / (batch_idx + 1)
    train_loss.append(total_train_loss)
    total_kld = total_kld / (batch_idx + 1)
    kld_loss.append(total_kld)

    scheduler.step()
    model.eval()

    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(val_dataloader):
            x = x.to(device)
            
            x_hat, _, _ = model(x)
            # x_hat, _, _, _ = model(x)
            if rv:
                loss = criterion(x, x_hat[:,0,...])
            else:
                loss = criterion(x, x_hat)
            total_val_loss += loss.item()
    total_val_loss = total_val_loss / (batch_idx + 1)
    val_loss.append(total_val_loss)
            
    print('\nEpoch: {}/{}, Train Loss: {:.8f}, KLD Loss: {:.8f}, Val Loss: {:.8f}'.format(epoch + 1, epochs, total_train_loss, total_kld, total_val_loss))
    if save_model:
        if total_val_loss < best_val_loss:
            best_val_loss =  total_val_loss
            print("Saving the model state dictionary for Epoch: {} with Validation loss: {:.8f}".format(epoch + 1, total_val_loss))
            torch.save(model.state_dict(), model_name)

fig=plt.figure(figsize=(10, 5))
plt.plot(np.arange(1, epochs+1), train_loss, label="Train loss")
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.title("Train Loss Plot")
# plt.legend(loc='upper right')
plt.show()

fig=plt.figure(figsize=(10, 5))
plt.plot(np.arange(1, epochs+1), kld_loss, label="KLD loss")
plt.xlabel('Epochs')
plt.ylabel('KLD Loss')
plt.title("KLD Loss Plot")
# plt.legend(loc='upper right')
plt.show()

fig=plt.figure(figsize=(10, 5))
plt.plot(np.arange(1, epochs+1), val_loss, label="Validation loss")
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.title("Validation Loss Plot")
# plt.legend(loc='upper right')
plt.show()

import pickle
if rv:
    with open(exp_name + "train_losses_rv.txt", "wb") as fp:
        tr_rv = pickle.dump(train_loss, fp)
    with open(exp_name + "kld_losses_rv.txt", "wb") as fp:
        tr_rv = pickle.dump(kld_loss, fp)
    with open(exp_name + "val_losses_rv.txt", "wb") as fp:
        val_rv = pickle.dump(val_loss, fp)
else:
    with open(exp_name + "train_losses.txt", "wb") as fp:
        tr = pickle.dump(train_loss, fp)
    with open(exp_name + "kld_losses.txt", "wb") as fp:
        tr_rv = pickle.dump(kld_loss, fp)
    with open(exp_name + "val_losses.txt", "wb") as fp:
        val = pickle.dump(val_loss, fp)

show_image(torchvision.utils.make_grid(x))
if rv:
    show_image(torchvision.utils.make_grid(x_hat[:,0,...]))
    # print('MEAN',torch.mean(x_hat[:,1,...]))
else:
    show_image(torchvision.utils.make_grid(x_hat))

range1 = torch.arange(-2,2,0.2)
range2 = torch.arange(-2,2,0.2)
means = torch.cartesian_prod(range1, range2)
if latent_dim > 2:
    means = torch.cat((means, torch.zeros((means.shape[0], latent_dim-2))), 1)
if rv:
    means = means.unsqueeze(1).to(device)
    out_var = torch.zeros((means.shape[0], 1, latent_dim)).to(device)
    z = torch.cat((means, out_var), 1)
else:
    z = means.to(device)

with torch.no_grad():
    generated_images = model.decode(z)
    print(generated_images.shape)
    if rv:
        show_image(torchvision.utils.make_grid(generated_images[:,0,...],range1.shape[0]))
    else:
        show_image(torchvision.utils.make_grid(generated_images,range1.shape[0]))
    # show_image(generated_images[:,0,...])