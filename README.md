# RV-VAE

[VIPriors ICCV Workshop 2023] Pytorch implementation of RV modules and VAE modifications from the paper: "RV-VAE: Integrating Random Variable Algebra into Variational Autoencoders"

[**Paper**](http://users.ics.forth.gr/~argyros/mypapers/2023_10_VIPRIORS_Nicodemou.pdf) | 
[**Citation**](#citation) 

![RV-VAE Concept Figure](https://github.com/VassilisCN/RV-VAE/blob/main/RV-VAE%20Concept%20Figure.png)

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)

## Installation

You can install the whole project via Docker.

To build the image run:
```bash
docker build -t rv_vae:latest .
```
To create and run the container interactively:
```bash
docker run -it --gpus "all" --privileged --ipc "host" --name rv_vae_cont rv_vae
```
If the container is stopped, you can start and use it interactively like so:  
```bash
docker start rv_vae_cont
docker exec -it -u root rv_vae_cont bash
```

Alternatively, you can install everything in the `requirements.txt` with Python 3.8 in order to run the project locally. 

## Usage

### Random Variable Modules

In order to use any of the `random_variable_modules`, you can edit your `PYTHONPATH` environment variable to include the modules folder and then import them in your code.

All `random_variable_modules` take as input `x` a Pytorch tensor of shape `(B, 2, ...)`, where `B` is the batch size. Also, they accept the same arguments as their non-RV counterparts. For specific shape sizes and outputs, you can see the comments in each module.

For examples of how to use them, you can refer to each of the individual [VAE-based architecture](https://github.com/VassilisCN/RV-VAE/tree/main/VAEs) modifications. Specifically, if you find in the code the flag `rv`, the `if` statement containing this flag changes the specific part of the code from the original to the RV-aware one.

### RV-VAEs

Each VAE architecture is used as each of their original authors suggested respectively. For each architecture, you will also find their respective README files for reference. All architectures are set to train in RV-aware mode.

You can start the training procedure in each architecture like so:

#### Soft-Intro-VAE

```bash
python3 RV_VAE/VAEs/Soft_Intro_VAE/main.py
```

#### DC-VAE

```bash
python3 RV_VAE/VAEs/DC_VAE/train.py
```

#### Vanilla VAE

```bash
python3 RV_VAE/VAEs/Vanilla_VAE/train.py
```
You can check each of these three files individually to change any parameters/datasets.

## Citation

If you include in your research Random Variable Modules or any parts of this project, please cite each respective work, if applicable, as well as this work:
```
@inproceedings{Nicodemou023,
  author = {Nicodemou, Vassilis C and Oikonomidis, Iason and Argyros, Antonis},
  title = {RV-VAE: Integrating Random Variable Algebra into Variational Autoencoders},
  booktitle = {International Conference on Computer Vision Workshops (ViPriors 2023 - ICCVW 2023), (to appear)},
  publisher = {IEEE},
  year = {2023},
  month = {October},
  address = {Paris, France},
  projects =  {VMWARE,I.C.HUMANS},
  pdflink = {http://users.ics.forth.gr/ argyros/mypapers/2023_10_VIPRIORS_Nicodemou.pdf}
}
```
