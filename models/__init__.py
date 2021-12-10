# Expose the most commonly-used deep nets directly from models for convenience:
from models.stylegan2.networks import Generator
from models.spatial_transformers.spatial_transformer import get_stn, ComposedSTN, SpatialTransformer
from models.losses.lpips import get_perceptual_loss
from models.losses.loss import gangealing_loss, gangealing_cluster_loss, total_variation_loss, flow_identity_loss, assign_fake_images_to_clusters
from models.spatial_transformers.antialiased_sampling import BilinearDownsample
from models.cluster_classifier import ResnetClassifier
from models.latent_learner import DirectionInterpolator, PCA, kmeans_plusplus
import torch


# Various convenience functions useful for training or inference:

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def batched_forward(net, input, device, batch_size=600, **kwargs):
    out = []
    for i in range(0, input.size(0), batch_size):
        x = input[i:i+batch_size].to(device)
        out.append(net(x, **kwargs).cpu())
    out = torch.cat(out, dim=0)
    return out


@torch.inference_mode()
def accuracy(predictions, gt_probabilities, k=1):
    # This is a sort of "reverse" top-K accuracy, where we see if the classifier's argmax prediction
    # was in the top-K "best" classes according to the ground-truth probabilities.
    top_pred = predictions.argmax(dim=1).unsqueeze(1)
    top_gt = gt_probabilities.topk(k=k, dim=1).indices
    acc_k = (top_pred == top_gt).any(dim=1).float().mean()
    return acc_k
