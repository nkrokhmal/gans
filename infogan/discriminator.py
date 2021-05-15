from imports.external import *


class Discriminator(nn.Module):
    def __init__(self, classes, channels, img_size, latent_dim, code_dim):
        super(Discriminator, self).__init__()
        self.classes = classes
        self.channels = channels
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.code_dim = code_dim
        self.img_shape = (self.channels, self.img_size, self.img_size)
        self.model = nn.Sequential(
            *self._create_conv_layer(self.channels, 16, True, False),
            *self._create_conv_layer(16, 32, True, True),
            *self._create_conv_layer(32, 64, True, True),
            *self._create_conv_layer(64, 128, True, True),
        )
        out_linear_dim = 128 * (self.img_size // 16) * (self.img_size // 16)
        self.adv_linear = nn.Linear(out_linear_dim, 1)
        self.class_linear = nn.Sequential(
            nn.Linear(out_linear_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, self.classes)
         )
        self.code_linear = nn.Sequential(
            nn.Linear(out_linear_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, self.code_dim)
        )
        self.adv_loss = torch.nn.MSELoss()
        self.class_loss = torch.nn.CrossEntropyLoss()
        self.style_loss = torch.nn.MSELoss()

    def _create_conv_layer(self, size_in, size_out, drop_out=True, normalize=True):
        layers = [nn.Conv2d(size_in, size_out, kernel_size=3, stride=2, padding=1)]
        if drop_out:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout(0.4))

        if normalize:
            layers.append(nn.BatchNorm2d(size_out, 0.8))
        return layers

    def forward(self, image):
        y_img = self.model(image)
        y_vec = y_img.view(y_img.shape[0], -1)
        y = self.adv_linear(y_vec)
        label = F.softmax(self.class_linear(y_vec), dim=1)
        code = self.code_linear(y_vec)
        return y, label, code



