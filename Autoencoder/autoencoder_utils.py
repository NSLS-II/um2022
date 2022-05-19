import torch


# Model
class Encoder(torch.nn.Module):# torch.nn.Module is the base class for NN

    def __init__(self, in_size, kernel, channels, ld):

        super( Encoder , self ).__init__() # get access to the methods of the parent class
        self.conv_layer = torch.nn.Conv2d(1, channels, kernel, stride=1, padding=(kernel-1)//2)
        self.flat_size = in_size * in_size * channels
        self.encoded = torch.nn.Linear(self.flat_size, ld)

    def forward(self, x):

        x = self.conv_layer(x) # get conv kernel map
        x = torch.nn.functional.relu(x) # apply activation function
        batchsize , features , nX , nY = x.size() # get shape of the convolution maps
        x = self.encoded(x.reshape( batchsize , 1 , features * nX * nY ) ) # compress to encoded state

        return x

class Decoder(torch.nn.Module):

    def __init__(self, out_size, kernel, channels, ld):

        super( Decoder , self ).__init__()
        self.linear = torch.nn.Linear(ld, out_size*out_size*channels)
        self.conv_layer = torch.nn.ConvTranspose2d(channels, 1, kernel, stride = 1, padding= (kernel-1)//2)
        self.out_size = out_size
        self.channels = channels

    def forward(self, x):

        x = self.linear(x) # convert to large number of variables
        x = x.reshape(x.size()[0], self.channels, self.out_size, self.out_size)
        x = self.conv_layer(x) # transpose conv transformation
        return x


class Autoencoder(torch.nn.Module):
    def __init__(self, in_size, kernel, channels, ld):
        super( Autoencoder , self ).__init__()
        self.kernel = kernel
        self.channels = channels
        self.in_size = in_size

        self.encoder = Encoder(self.in_size, self.kernel, self.channels, ld)
        self.decoder = Decoder(self.in_size, self.kernel, self.channels, ld)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def get_latent_space(self, x):
        return self.encoder(x)



def create_example(p):
    x = torch.rand(size = (p, p))
    diag_x= torch.zeros_like(x)
    diag_x.fill_diagonal_(3)
    x = x + diag_x

    x = (x - x.min())/(x.max()- x.min())

    return x

def generate_dataset(N, p):
    D = torch.empty((N, 1, p, p))
    for j in range(N):
        D[j, 0, :, :] = create_example(p)

    return D

def generate_abnormal_1(p, with_noise = False):
    y = torch.zeros((p,p))
    y[:2, p-1] = 1
    y[-2:, 0] = 1
    if with_noise:
        y += 0.33*torch.rand(size = (p, p))
        y = (y - y.min())/(y.max()- y.min())
    return y

def generate_abnormal_2(p, with_noise = False):
    y = torch.zeros((p,p))
    y[1, 1] = 1
    y[1, 3] = 1
    y[4, 1] = 1
    y[4, 3] = 1
    y[3, 2] = 1
    if with_noise:
        y += 0.33*torch.rand(size = (p, p))
        y = (y - y.min())/(y.max()- y.min())
    return y
