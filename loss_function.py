from torch import nn
import torch

class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, model_output, targets):
        mel_target, gate_target, f0_target = targets[0], targets[1], targets[2]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        f0_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)
        f0_target = f0_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, f0_output, _ = model_output
        gate_out = gate_out.view(-1, 1)
        f0_output = f0_output.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        f0_loss = nn.L1Loss()(f0_output, f0_target)
        return 0.75* mel_loss + gate_loss + 0.25 * f0_loss

# This cannot be used unless initial f0 predictions are not zero.
# Since initial F0 predictions are all zero, GPE is zero for initial stage.
class GPELoss(nn.Module):
    def __init__(self):
        super(GPELoss, self).__init__()

    def forward(self, f0_out, f0_target):
        out_voiced_mask = f0_out != 0
        tmp1 = out_voiced_mask.cpu().numpy()
        target_voiced_mask = f0_target != 0
        tmp2 = target_voiced_mask.cpu().numpy()
        diff_abs = (f0_out - f0_target).abs()
        # tmp3 = diff_abs.cpu().numpy()
        erronous_prediction_mask = diff_abs > 0.2 * f0_target
        tmp4 = erronous_prediction_mask.cpu().numpy()

        denumerator = out_voiced_mask * target_voiced_mask * erronous_prediction_mask
        tmp5 = denumerator.cpu().numpy()
        numerator = out_voiced_mask * target_voiced_mask
        tmp6 = numerator.cpu().numpy()

        denumerator = denumerator.sum()
        numerator = numerator.sum()
        loss = denumerator / (numerator+1e-3)
        return loss

class FFELoss(nn.Module):
    def __init__(self):
        super(FFELoss, self).__init__()

    def forward(self, f0_out, f0_target):
        out_voiced_mask = f0_out != 0
        tmp1 = out_voiced_mask.cpu().numpy()
        target_voiced_mask = f0_target != 0
        tmp2 = target_voiced_mask.cpu().numpy()
        diff_abs = (f0_out - f0_target).abs()
        # tmp3 = diff_abs.cpu().numpy()
        erronous_prediction_mask = diff_abs > 0.2 * f0_target
        tmp4 = erronous_prediction_mask.cpu().numpy()

        numerator = torch.cuda.FloatTensor([f0_target.shape[0]])
        denumerator1 = out_voiced_mask * target_voiced_mask * erronous_prediction_mask
        denumerator1 = denumerator1.sum()
        denumerator2 = out_voiced_mask != target_voiced_mask
        denumerator2 = denumerator2.sum()
        denumerator = torch.cuda.FloatTensor([denumerator1 + denumerator2])
        loss = denumerator / (numerator) # removed adding 1e-3 to numerator because it seems unlikely for numerator to be zero
        return loss


class Tacotron2Loss_GPE(nn.Module):
    def __init__(self):
        super(Tacotron2Loss_GPE, self).__init__()

    def forward(self, model_output, targets):
        mel_target, gate_target, f0_target = targets[0], targets[1], targets[2]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        f0_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)
        f0_target = f0_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, f0_output, _ = model_output
        gate_out = gate_out.view(-1, 1)
        f0_output = f0_output.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        f0_loss = GPELoss()(f0_output, f0_target)
        return mel_loss + gate_loss + f0_loss

class Tacotron2Loss_FFE(nn.Module):
    def __init__(self):
        super(Tacotron2Loss_FFE, self).__init__()

    def forward(self, model_output, targets):
        mel_target, gate_target, f0_target = targets[0], targets[1], targets[2]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        f0_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)
        f0_target = f0_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, f0_output, _ = model_output
        gate_out = gate_out.view(-1, 1)
        f0_output = f0_output.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        f0_loss = FFELoss()(f0_output, f0_target)
        return mel_loss + gate_loss + 10* f0_loss