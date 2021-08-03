import numpy as np
from scipy.stats import entropy
from torch import nn
from torch.nn import functional as F
from torchvision.models.inception import inception_v3


class InceptionScore(nn.Module):
    def __init__(self, device, batch_size=32, resize=False):
        super(InceptionScore, self).__init__()
        assert batch_size > 0
        self.resize = resize
        self.batch_size = batch_size
        self.device = device
        # Load inception model
        self.inception_model = inception_v3(pretrained=True, transform_input=False).to(self.device)
        self.inception_model.eval()
        self.clean()

    def clean(self):
        self.preds = np.zeros((0, 1000))

    def get_pred(self, x):
        if self.resize:
            x = F.interpolate(x, size=(299, 299), mode='bilinear',
                              align_corners=True)  # leaving the before 0.4.0 default
        x = self.inception_model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()

    def forward(self, imgs):
        # Get predictions
        preds_imgs = self.get_pred(imgs.to(self.device))
        self.preds = np.append(self.preds, preds_imgs, axis=0)

    def compute_score(self, splits=1):
        # Now compute the mean kl-div
        split_scores = []
        preds = self.preds
        N = self.preds.shape[0]
        for k in range(splits):
            part = preds[k * (N // splits): (k + 1) * (N // splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))

        return np.mean(split_scores), np.std(split_scores)

