from imports.external import *
from infogan.upsample import Upsample


class Generator(nn.Module):
    def __init__(self, classes, channels, img_size, latent_dim, code_dim):
        super(Generator, self).__init__()
        self.classes = classes
        self.channels = channels
        self.img_size = img_size
        self.img_init_shape = (128, self.img_init_shape, self.img_init_shape)
        self.img_shape = (self.channels, self.img_size, self.img_size)
        self.stem_linear = nn.Sequential(
            nn.Linear(latent_dim + classes + code_dim, int(np.prod(self.img_init_shape)))
        )
        self.model = nn.Sequential(
            nn.BatchNorm2d(128),
            *self._create_deconv_layer(128, 128, upsample=True),
            *self._create_deconv_layer(128, 64, upsample=True),
            *self._create_deconv_layer(64, self.channels, upsample=False, normalize=False),
            nn.Tanh()
        )

    def _create_deconv_layer(self, size_in, size_out, upsample=True, normalize=True):
        layers = []
        if upsample:
            layers.append(Upsample(scale_factor=2))
        layers.append((nn.Conv2d(size_in, size_out)))