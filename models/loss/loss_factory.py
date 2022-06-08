import torch
import torch.nn as nn


def _make_input(t, requires_grad=False, device=torch.device('cpu')):
    """Make zero inputs for AE loss.

    Args:
        t (torch.Tensor): input
        requires_grad (bool): Option to use requires_grad.
        device: torch device

    Returns:
        torch.Tensor: zero input.
    """
    inp = torch.autograd.Variable(t, requires_grad=requires_grad)
    inp = inp.sum()
    inp = inp.to(device)

    return inp


class HeatmapsLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, preds, targets, masks=None):
        total_loss = 0
        for i, (pred, target) in enumerate(zip(preds, targets)):
            loss = self.criterion(pred, target)
            if masks is not None:
                loss = loss * masks[i][:, None, :, :]
            total_loss += loss.mean(dim=3).mean(dim=2).mean(dim=1).mean(dim=0)

        return total_loss


class AssociativeEmbeddingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def singleTagLoss(self, pred_tag, joints):
        """Associative embedding loss for one image.

        Note:
            - heatmaps weight: W
            - heatmaps height: H
            - max_num_people: M
            - num_keypoints: K

        Args:
            pred_tag (torch.Tensor[KxHxW,1]): tag of output for one image.
            joints (torch.Tensor[M,K,2]): joints information for one image.
        """
        tags = []
        pull = 0
        for joints_per_person in joints:
            tmp = []
            for joint in joints_per_person:
                if joint[1] > 0:
                    tmp.append(pred_tag[joint[0]])
            if len(tmp) == 0:
                continue
            tmp = torch.stack(tmp)
            tags.append(torch.mean(tmp, dim=0))
            pull = pull + torch.mean((tmp - tags[-1].expand_as(tmp))**2)

        num_tags = len(tags)
        if num_tags == 0:
            return (
                _make_input(torch.zeros(1).float(), device=pred_tag.device),
                _make_input(torch.zeros(1).float(), device=pred_tag.device))
        elif num_tags == 1:
            return (_make_input(
                torch.zeros(1).float(), device=pred_tag.device), pull)

        tags = torch.stack(tags)

        size = (num_tags, num_tags)
        A = tags.expand(*size)
        B = A.permute(1, 0)

        diff = A - B
        not_diagonal = torch.eye(num_tags).to(diff.device) != 1
        diff = diff[not_diagonal]

        push = torch.exp(-(diff ** 2)).mean()
        push_loss = push * 0.5
        pull_loss = pull / num_tags

        return push_loss, pull_loss

    def forward(self, tags, joints):
        """Accumulate the tag loss for each image in the batch.

        Note:
            - batch_size: N
            - heatmaps weight: W
            - heatmaps height: H
            - max_num_people: M
            - num_keypoints: K

        Args:
            tags (torch.Tensor[N,K,H,W]): tag channels of output.
            joints (torch.Tensor[N,M,K,2]): joints information.
        """
        pushes, pulls = [], []
        joints = joints.cpu().data.numpy()
        batch_size = tags.size(0)
        tags = tags.reshape([batch_size, -1, 1])
        for i in range(batch_size):
            push, pull = self.singleTagLoss(tags[i], joints[i])
            pushes.append(push)
            pulls.append(pull)

        pull_loss = torch.stack(pulls, dim=0).mean()
        push_loss = torch.stack(pushes, dim=0).mean()

        return pull_loss, push_loss
