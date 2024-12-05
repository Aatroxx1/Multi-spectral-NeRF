from torchmetrics.functional.image.sam import spectral_angle_mapper


def sam_loss(pred, gt):
    pred = pred.unsqueeze(-1).unsqueeze(-1)
    gt = gt.unsqueeze(-1).unsqueeze(-1)
    return spectral_angle_mapper(pred, gt)
