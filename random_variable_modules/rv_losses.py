import torch


def Mahalanobis_classification_distance(random_variables, target):
    """
    random_variables: (N, 2, C) where N are the minibatches, C the number of classes
    target: (N)
    """
    N, C = random_variables.shape[0], 1 if len(random_variables.shape) == 2 else random_variables.shape[2]
    targets = torch.zeros(C, device=target.device)
    targets[target[0]] = 1
    targets = torch.unsqueeze(targets, 0)
    # Some optimization can be done to remove for loop an improve time
    for n in range(1, N):
        zeros = torch.zeros(C, device=target.device)
        zeros[target[n]] = 1
        zeros = torch.unsqueeze(zeros, 0)
        targets = torch.cat((targets, zeros), 0)
    # Now targets are of shape (N, C)
    return torch.sum((torch.sum(((targets - random_variables[:, 0, :]) ** 2) / (random_variables[:, 1, :] ** 2), 1)) ** 0.5) / N

def RandomVariableMSELoss(random_variables, target, lamda=55):
    """
    random_variables: (N, 2, *) where N are the minibatches
    target: (N, *)
    """
    random_var_mean = random_variables[:, 0, ...]
    random_var_vars = random_variables[:, 1, ...]
    MSE = torch.nn.MSELoss()
    return MSE(random_var_mean, target) + lamda*MSE(random_var_vars, torch.zeros(random_var_vars.shape, device=target.device))

def RandomVariableCE_MSELoss(random_variables, target, lamda=55):
    """
    random_variables: (N, 2, C) where N are the minibatches, C the number of classes 
    target: (N)
    """
    N, C = random_variables.shape[0], 1 if len(random_variables.shape) == 2 else random_variables.shape[2]
    random_var_mean = random_variables[:, 0, :]
    random_var_vars = random_variables[:, 1, :]
    CE = torch.nn.CrossEntropyLoss()
    MSE = torch.nn.MSELoss()
    return CE(random_var_mean, target) + lamda*MSE(random_var_vars, torch.zeros((N,C), device=target.device))

def Z_testLoss(random_variables, target):
    """
    random_variables: (N, 2, C) where N are the minibatches, C the number of classes 
    target: (N)
    """
    N, C = random_variables.shape[0], 1 if len(random_variables.shape) == 2 else random_variables.shape[2]
    batches = torch.tensor(range(N))
    targets = random_variables[batches,:,target]
    not_targets_ids = [[x for x in range(C) if x != c] for c in target]
    not_targets = random_variables[batches.unsqueeze(1),:,not_targets_ids]
    loss = 0
    for c in range(C-1):
        loss += (targets[:,0] - not_targets[:,c,0])/(targets[:,1] + not_targets[:,c,1])**0.5
    return -torch.sum(loss)/(N*(C-1))

def BhattacharyyaLoss(random_variables, target):
    """
    random_variables: (N, 2, C) where N are the minibatches, C the number of classes 
    target: (N)
    """
    N, C = random_variables.shape[0], 1 if len(random_variables.shape) == 2 else random_variables.shape[2]
    batches = torch.tensor(range(N))
    targets = random_variables[batches,:,target]
    not_targets_ids = [[x for x in range(C) if x != c] for c in target]
    not_targets = random_variables[batches.unsqueeze(1),:,not_targets_ids]
    targets = torch.unsqueeze(targets, 1)
    not_targets_mean = not_targets[..., 0]
    not_targets_vars = not_targets[..., 1]
    targets_mean = targets[..., 0]
    targets_vars = targets[..., 1]
    d1 = -(1/4)*torch.log((1/4)*((torch.true_divide((targets_vars**2),(not_targets_vars**2)))+(torch.true_divide((not_targets_vars**2),(targets_vars**2)))+2))- \
        ((1/4)*(torch.true_divide(((targets_mean-not_targets_mean)**2),((targets_vars**2)+(not_targets_vars**2))))*torch.sign(targets_mean-not_targets_mean))
    for c1 in range(C-2):
        for c2 in range(c1+1, C-1):
            d2 = (1/4)*torch.log((1/4)*((torch.true_divide((not_targets_vars[...,c1]**2),(not_targets_vars[...,c2]**2)))+(torch.true_divide((not_targets_vars[...,c2]**2),(not_targets_vars[...,c1]**2)))+2))+\
                ((1/4)*(torch.true_divide(((not_targets_mean[...,c1]-not_targets_mean[...,c2])**2),((not_targets_vars[...,c1]**2)+(not_targets_vars[...,c2]**2)))))
            d2 = d2.unsqueeze(1)
            d1 += d2
    return torch.sum(d1)/(N*(C-1))

def CustomLoss(random_variables, target):
    """
    random_variables: (N, 2, C) where N are the minibatches, C the number of classes 
    target: (N)
    """
    N, C = random_variables.shape[0], 1 if len(random_variables.shape) == 2 else random_variables.shape[2]
    batches = torch.tensor(range(N))
    targets = random_variables[batches,:,target]
    not_targets_ids = [[x for x in range(C) if x != c] for c in target]
    not_targets = random_variables[batches.unsqueeze(1),:,not_targets_ids]
    loss = 0
    for c in range(C-1):
        loss += (targets[:,0] - 2*targets[:,1]) - (not_targets[:,c,0] + 2*not_targets[:,c,1])
    targets2 = torch.ones(N, 2, C, device=target.device)*(-5)
    for n in range(N):
        targets2[n, 0, target[n]] = 2
        targets2[n, 1, target[n]] = 0.5
    return -torch.sum(loss)/(N*(C-1)) + torch.nn.functional.mse_loss(random_variables, targets2)


if __name__ == "__main__":
    torch.manual_seed(0)
    a = torch.rand(3,2,5)
    t = torch.tensor([1,2,3])
    loss = RandomVariableCE_MSELoss(a, t)
    print(a)
    print(loss)
