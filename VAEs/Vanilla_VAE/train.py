import torch, torchvision
from torchvision import transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from vaes_models import *
from vaes_losses import *
import numpy as np
dts = {0:'MNIST', 1:'CIFAR10', 2:'CelebA'}


#############################################
# Arguments
#############################################
seed = 123

dataset = 0 # Can choose from {0:'MNIST', 1:'CIFAR10', 2:'CelebA'}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

save_model = True
latent_dim = 2
lr = 0.005 # Learning rate
batch_size = 64
epochs = 4
exp_name = "VAE_{}".format(dts[dataset]) # Destination and name of saved model file
rv = True # Enable RV awarness
#############################################
# Arguments
#############################################

torch.manual_seed(seed)
np.random.seed(seed) 
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

def show_image(x, name=None):
    x = (x + 1) / 2
    fig = plt.figure()
    plt.imshow(np.transpose(x.cpu().numpy(), (1, 2, 0)))
    if name:
        plt.savefig('ims_{}.png'.format(name))
    else:
        plt.show()


if dataset==0:
    celeba_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Resize(64)]))
    celeba_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Resize(64)]))
elif dataset==1:
    celeba_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=data_transforms())
    celeba_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=data_transforms())
elif dataset==2:
    celeba_trainset = datasets.CelebA(root='./data', split='train', download=True, transform=data_transforms())
    celeba_testset = datasets.CelebA(root='./data', split='test', download=True, transform=data_transforms())
else:
    raise NotImplementedError("dataset is not supported")

train_dataloader = torch.utils.data.DataLoader(celeba_trainset, batch_size=batch_size, shuffle=True, drop_last=True)
val_dataloader = torch.utils.data.DataLoader(celeba_testset, batch_size=batch_size, shuffle=False, drop_last=True)
dataiter = iter(train_dataloader)
images, labels = next(dataiter)
# show_image(torchvision.utils.make_grid(images))

print("Training dataset size: ", len(celeba_trainset))
print("Validation dataset size: ", len(celeba_testset))

print(device)

if rv:
    model_name = exp_name + '_RV.pt'
else:
    model_name = exp_name + '.pt'

# torch.autograd.set_detect_anomaly(True)
model = VanillaVAE(in_channels=images[0].shape[0], latent_dim=latent_dim, rv=rv).to(device)

loss_function = VanillaLoss

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
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
        x_hat, mean, log_var = model(x)

        loss, kld = loss_function(x, x_hat, mean, log_var, celeba_trainset, rv)

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
if rv:
    plt.savefig('train_rec_rv.png')
else:
    plt.savefig('train_rec.png')

fig=plt.figure(figsize=(10, 5))
plt.plot(np.arange(1, epochs+1), kld_loss, label="KLD loss")
plt.xlabel('Epochs')
plt.ylabel('KLD Loss')
plt.title("KLD Loss Plot")
# plt.legend(loc='upper right')
if rv:
    plt.savefig('train_kld_rv.png')
else:
    plt.savefig('train_kld.png')

fig=plt.figure(figsize=(10, 5))
plt.plot(np.arange(1, epochs+1), val_loss, label="Validation loss")
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.title("Validation Loss Plot")
# plt.legend(loc='upper right')
if rv:
    plt.savefig('val_loss_rv.png')
else:
    plt.savefig('val_loss.png')

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

show_image(torchvision.utils.make_grid(x), name='input')
if rv:
    show_image(torchvision.utils.make_grid(x_hat[:,0,...]), name='recs')
else:
    show_image(torchvision.utils.make_grid(x_hat), name='recs')

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
        show_image(torchvision.utils.make_grid(generated_images[:,0,...],range1.shape[0]), name='gens')
    else:
        show_image(torchvision.utils.make_grid(generated_images,range1.shape[0]), name='gens')