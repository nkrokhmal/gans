from imports.external import *
from infogan.upsample import Upsample


class Generator(nn.Module):
    def __init__(self, classes, channels, img_size, latent_dim, code_dim):
        super(Generator, self).__init__()
        self.classes = classes
        self.channels = channels
        self.img_size = img_size
        self.img_init_size = self.img_size // 4
        self.latent_dim = latent_dim
        self.code_dim = code_dim
        self.img_init_shape = (128, self.img_init_size, self.img_init_size)
        self.img_shape = (self.channels, self.img_size, self.img_size)
        self.stem_linear = nn.Sequential(
            nn.Linear(latent_dim + classes + code_dim,
                      int(np.prod(self.img_init_shape)))
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
        layers.append(nn.Conv2d(size_in, size_out, 3, stride=1, padding=1))
        if normalize:
            layers.append(nn.BatchNorm2d(size_out, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers

    def forward(self, noise, labels, code):
        z = torch.cat((noise, labels, code), -1)
        z_vec = self.stem_linear(z)
        z_img = z_vec.view(z_vec.shape[0], *self.img_init_shape)
        x = self.model(z_img)
        return x