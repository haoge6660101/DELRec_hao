import torch
import torch.nn as nn
import torch.optim as optim
import torch
from data.utils import calculate_metrics

class MultiTaskModel(nn.Module):
    def __init__(self, input_dim):
        super(MultiTaskModel, self).__init__()
        self.shared_layer = nn.Linear(input_dim, 64)
        self.task1_output = nn.Linear(64, 1)
        self.task2_output = nn.Linear(64, 1)
        self.log_var_task1 = nn.Parameter(torch.zeros((1,), requires_grad=True))
        self.log_var_task2 = nn.Parameter(torch.zeros((1,), requires_grad=True))

    def forward(self, x):
        shared_representation = torch.relu(self.shared_layer(x))
        task1_output = torch.sigmoid(self.task1_output(shared_representation))
        task2_output = self.task2_output(shared_representation)
        return task1_output, task2_output, self.log_var_task1, self.log_var_task2



def dynamic_loss_weighting(task1_loss, task2_loss, parameters, alpha=0.5):

    task1_loss_ratio = task1_loss / task1_loss.detach()
    task2_loss_ratio = task2_loss / task2_loss.detach()

    task1_grad_norm = torch.norm(torch.autograd.grad(task1_loss, parameters, retain_graph=True)[0])
    task2_grad_norm = torch.norm(torch.autograd.grad(task2_loss, parameters, retain_graph=True)[0])


    task1_train_rate = task1_loss_ratio / task1_grad_norm
    task2_train_rate = task2_loss_ratio / task2_grad_norm

    task1_weight = (task1_train_rate ** alpha) / (task1_train_rate ** alpha + task2_train_rate ** alpha)
    task2_weight = (task2_train_rate ** alpha) / (task1_train_rate ** alpha + task2_train_rate ** alpha)

    loss = task1_weight * task1_loss + task2_weight * task2_loss
    return loss

