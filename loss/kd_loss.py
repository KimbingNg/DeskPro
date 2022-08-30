# Borrowed from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py

from torch import nn
from torch.nn import functional as F


def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs and labels.
    Args:
        outputs: (Variable) dimension batch_size x 6 - output of the model
        labels: (Variable) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]
    Returns:
        loss (Variable): cross entropy loss for all images in the batch
    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """
    return nn.CrossEntropyLoss()(outputs, labels)


def loss_fn_kd(outputs, teacher_outputs, T):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    KD_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs / T, dim=1),
                             F.softmax(teacher_outputs.detach() / T, dim=1)) * (T * T)
    # + \
    # F.cross_entropy(outputs, labels) * (1. - alpha)
    return KD_loss


class KDLoss(nn.Module):
    def __init__(self, T):
        super(KDLoss, self).__init__()
        self.T = T

    def forward(self, lr_face_out, face_out):
        loss = 0
        for logit_student, logit_teacher in zip(lr_face_out[0], face_out[0]):
            loss += loss_fn_kd(logit_student, logit_teacher, self.T)
        return loss
