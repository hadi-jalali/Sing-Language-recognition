from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet101, vgg11


class ConvBlock(nn.Module):
    """Convolution block used in CNN."""
    def __init__(self, 
                 channel_in, 
                 channel_out, 
                 activation_fn, 
                 use_batchnorm, 
                 pool:str='max_2',
                 kernel_size:int=3):
        
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(channel_in, channel_out, kernel_size)
        self.conv2 = nn.Conv2d(channel_out, channel_out, kernel_size)

        self.use_batchnorm = use_batchnorm
        if self.use_batchnorm:
            self.batchnorm1 = nn.BatchNorm2d(channel_out, momentum=0.01)
            self.batchnorm2 = nn.BatchNorm2d(channel_out, momentum=0.01)
        
        if activation_fn == "relu":
            self.a_fn = nn.ReLU()
        elif activation_fn == "leaky_relu":
            self.a_fn = nn.LeakyReLU()
        elif activation_fn == "param_relu":
            self.a_fn = nn.PReLU()
        else:
            raise ValueError("please use a valid activation function argument ('relu'; 'leaky_relu'; 'param_relu')")

        if pool == "max_2":
            self.pool = nn.MaxPool2d(2)
        elif pool == "adap":
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif not pool:
            self.pool = pool
        else:
            raise ValueError("please use a valid pool argument ('max_2', 'adap', None)")
    
    def forward(self, x):
        out = self.conv1(x)
        if self.use_batchnorm:
            out = self.batchnorm1(out)
        out = self.a_fn(out)

        out = self.conv2(out)
        if self.use_batchnorm:
            out = self.batchnorm2(out)
        out = self.a_fn(out)
        if self.pool:
            out = self.pool(out)
        return out

class CNN_Encoder(nn.Module):
    """CNN Encoder for CNN part of model."""
    def __init__(self, 
                 channel_out, 
                 n_layers, 
                 intermediate_act_fn="relu", 
                 use_batchnorm=True, 
                 channel_in=3):
        
        super(CNN_Encoder, self).__init__()

        channels = [64, 128, 256, 512]

        if n_layers < 1 or n_layers > len(channels)*2:
            raise ValueError(f"please use a valid int for the n_layers param (1-{len(channels)*2})")
        
        n_repeat = remainder = max(0, n_layers - len(channels))
        pointer = 0

        self.conv1 = nn.Conv2d(channel_in, channels[0], 3, stride=2)
        self.maxpool = nn.MaxPool2d(3, 2)

        layers =  OrderedDict()

        if n_layers > 1:
            layers[str(0)] = ConvBlock(channels[0], channels[0], intermediate_act_fn, use_batchnorm=use_batchnorm)

        for i in range(1, n_layers-1):
            if i % 2 == 0 and remainder > 0:
                layers[str(i)] = ConvBlock(channels[pointer], channels[pointer], intermediate_act_fn, use_batchnorm=use_batchnorm, pool=None)
                remainder -= 1
            else:
                layers[str(i)] = ConvBlock(channels[pointer], channels[min(pointer+1, len(channels)-1)], intermediate_act_fn, use_batchnorm=use_batchnorm)
                pointer += 1

        self.layers = nn.Sequential(layers)
        if n_layers < len(channels):
            conv_to_fc = channels[n_layers-1-n_repeat]
        else:
            conv_to_fc = channels[-1]
        
        self.conv2 = ConvBlock(channels[n_layers-2-n_repeat], conv_to_fc, intermediate_act_fn, use_batchnorm=use_batchnorm, pool="adap")
        
        self.fc = nn.Linear(conv_to_fc, channel_out)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.layers(out)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

class LSTM_Decoder(nn.Module):
    """LSTM Decoder for RNN part of model."""
    def __init__(self, 
                 embed_dim, 
                 hidden_dim, 
                 channel_out, 
                 n_layers, 
                 intermediate_act_fn="relu", 
                 bidirectional=False, 
                 attention=False,
                 device="cuda"):
        
        super(LSTM_Decoder, self).__init__()

        self.num_layers = n_layers
        self.hidden_dim = hidden_dim
        self.device = device
        self.bidirectional = bidirectional
        self.attention = attention

        self.lstm = nn.LSTM(embed_dim, 
                            self.hidden_dim, 
                            num_layers=self.num_layers, 
                            bidirectional=self.bidirectional, 
                            batch_first=True)

        if self.bidirectional:
            fc1_in = self.hidden_dim * 2
        else:
            fc1_in = self.hidden_dim
        self.fc1 =  nn.Linear(fc1_in, channel_out)

        if intermediate_act_fn == "relu":
            self.a_fn = nn.ReLU()
        elif intermediate_act_fn == "leaky_relu":
            self.a_fn = nn.LeakyReLU()
        elif intermediate_act_fn == "param_relu":
            self.a_fn = nn.PReLU()
        else:
            raise ValueError("please use a valid activation function argument ('relu'; 'leaky_relu'; 'param_relu')")
        
        if self.attention:
            self.attention_layer = nn.Linear(2 * self.hidden_dim if self.bidirectional else self.hidden_dim, 1)

    def forward(self, x):
        
        if self.bidirectional:
            h_0_size = c_0_size = self.num_layers * 2
        else:
            h_0_size = c_0_size = self.num_layers
        h = torch.zeros(h_0_size, x.size(0), self.hidden_dim).to(self.device)
        c = torch.zeros(c_0_size, x.size(0), self.hidden_dim).to(self.device)

        self.lstm.flatten_parameters()

        # Propagate input through LSTM
        out, (h, c) = self.lstm(x, (h, c)) #lstm with input, hidden, and internal state
        
        if self.attention:
            attention_w = F.softmax(self.attention_layer(out).squeeze(-1), dim=-1)
            
            out = torch.sum(attention_w.unsqueeze(-1) * out, dim=1)
        else:
            out = out[:, -1, :]
            
        out = self.fc1(out)
        
        return out, h

class CNN_LSTM(nn.Module):
    """CNN-LSTM model combining CNN and LSTM components."""
    def __init__(self, 
                 n_classes, 
                 latent_size, 
                 n_cnn_layers, 
                 n_rnn_layers, 
                 n_rnn_hidden_dim,
                 channel_in=3, 
                 cnn_act_fn="relu", 
                 rnn_act_fn="relu", 
                 dropout_rate=0.1,
                 cnn_bn=False, 
                 bidirectional=False, 
                 attention=False,
                 device="cuda"):
        
        super(CNN_LSTM, self).__init__()

        self.attention = attention

        self.encoder = CNN_Encoder(latent_size, 
                                   n_cnn_layers, 
                                   intermediate_act_fn=cnn_act_fn, 
                                   use_batchnorm=cnn_bn, 
                                   channel_in=channel_in)
        self.decoder = LSTM_Decoder(latent_size, 
                                    n_rnn_hidden_dim, 
                                    n_classes, 
                                    n_rnn_layers, 
                                    intermediate_act_fn=rnn_act_fn, 
                                    bidirectional=bidirectional, 
                                    attention=attention,
                                    device=device)

        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()

        cnn_in = x.view(batch_size * timesteps, C, H, W)
        latent_var = self.encoder(cnn_in)

        rnn_in = latent_var.view(batch_size, timesteps, -1)
        rnn_in = self.dropout(rnn_in)
        out, _ = self.decoder(rnn_in)

        return out

class VGG_LSTM(nn.Module):
    """VGG-LSTM model using VGG11 as CNN component."""
    def __init__(self, 
                 n_classes, 
                 latent_size, 
                 n_rnn_layers,
                 n_rnn_hidden_dim,
                 rnn_act_fn="relu",
                 dropout_rate=0.1,
                 bidirectional=False):
        
        super(VGG_LSTM, self).__init__()
        
        self.CNN = vgg11(pretrained=True, progress=False)
        self.CNN.classifier[6] = nn.Linear(4096, latent_size)

        self.decoder = LSTM_Decoder(latent_size, n_rnn_hidden_dim, n_classes, n_rnn_layers, intermediate_act_fn=rnn_act_fn, bidirectional=bidirectional)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()

        cnn_in = x.view(batch_size * timesteps, C, H, W)
        latent_var = self.CNN(cnn_in)

        rnn_in = latent_var.view(batch_size, timesteps, -1)
        rnn_in = self.dropout(rnn_in)
        out = self.decoder(rnn_in)

        return out

# extra models for experimentation below

class GRU_Decoder(nn.Module):

    def __init__(self, 
                 embed_dim, 
                 hidden_dim, 
                 channel_out, 
                 n_layers, 
                 intermediate_act_fn="relu", 
                 bidirectional=False, 
                 device="cuda"):
        
        super(GRU_Decoder, self).__init__()

        self.num_layers = n_layers
        self.hidden_dim = hidden_dim
        self.device = device
        self.bidirectional = bidirectional

        self.gru = nn.GRU(embed_dim, 
                            self.hidden_dim, 
                            num_layers=self.num_layers, 
                            bidirectional=self.bidirectional, 
                            batch_first=True)

        if self.bidirectional:
            fc1_in = self.hidden_dim * 2
        else:
            fc1_in = self.hidden_dim
        self.fc1 =  nn.Linear(fc1_in, channel_out)

        if intermediate_act_fn == "relu":
            self.a_fn = nn.ReLU()
        elif intermediate_act_fn == "leaky_relu":
            self.a_fn = nn.LeakyReLU()
        elif intermediate_act_fn == "param_relu":
            self.a_fn = nn.PReLU()
        else:
            raise ValueError("please use a valid activation function argument ('relu'; 'leaky_relu'; 'param_relu')")

    def forward(self, x):
        
        if self.bidirectional:
            h_0_size = self.num_layers * 2
        else:
            h_0_size = self.num_layers
        h = torch.zeros(h_0_size, x.size(0), self.hidden_dim).to(self.device)

        self.gru.flatten_parameters()

        # Propagate input through LSTM
        output, h = self.gru(x, h) #lstm with input, hidden, and internal state
        
        out = output[:, -1, :]

        out = self.fc1(out)
        
        return out

class RNN_Decoder(nn.Module):

    def __init__(self, 
                 embed_dim, 
                 hidden_dim, 
                 channel_out, 
                 n_layers, 
                 intermediate_act_fn="relu", 
                 bidirectional=False, 
                 attention=False,
                 device="cuda"):
        
        super(RNN_Decoder, self).__init__()

        self.num_layers = n_layers
        self.hidden_dim = hidden_dim
        self.device = device
        self.bidirectional = bidirectional
        self.attention = attention

        self.rnn = nn.RNN(embed_dim, 
                            self.hidden_dim, 
                            num_layers=self.num_layers, 
                            bidirectional=self.bidirectional, 
                            batch_first=True)

        if self.bidirectional:
            fc1_in = self.hidden_dim * 2
        else:
            fc1_in = self.hidden_dim
        self.fc1 =  nn.Linear(fc1_in, channel_out)

        if intermediate_act_fn == "relu":
            self.a_fn = nn.ReLU()
        elif intermediate_act_fn == "leaky_relu":
            self.a_fn = nn.LeakyReLU()
        elif intermediate_act_fn == "param_relu":
            self.a_fn = nn.PReLU()
        else:
            raise ValueError("please use a valid activation function argument ('relu'; 'leaky_relu'; 'param_relu')")

    def forward(self, x):
        
        if self.bidirectional:
            h_0_size = self.num_layers * 2
        else:
            h_0_size = self.num_layers
        h = torch.zeros(h_0_size, x.size(0), self.hidden_dim).to(self.device)

        self.rnn.flatten_parameters()
        
        # Propagate input through LSTM
        out, h = self.rnn(x, h) #lstm with input, hidden, and internal state
        
        if not self.attention:
            out = out[:, -1, :]
            out = self.fc1(out)
        
        return out, h

class CNN_GRU(nn.Module):

    def __init__(self, 
                 n_classes, 
                 latent_size, 
                 n_cnn_layers, 
                 n_rnn_layers, 
                 n_rnn_hidden_dim,
                 channel_in=3, 
                 cnn_act_fn="relu", 
                 rnn_act_fn="relu", 
                 dropout_rate=0.1,
                 cnn_bn=False, 
                 bidirectional=False):
        
        super(CNN_GRU, self).__init__()

        self.encoder = CNN_Encoder(latent_size, n_cnn_layers, intermediate_act_fn=cnn_act_fn, use_batchnorm=cnn_bn, channel_in=channel_in)
        self.decoder = GRU_Decoder(latent_size, n_rnn_hidden_dim, n_classes, n_rnn_layers, intermediate_act_fn=rnn_act_fn, bidirectional=bidirectional)

        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):

        batch_size, timesteps, C, H, W = x.size()

        cnn_in = x.view(batch_size * timesteps, C, H, W)
        latent_var = self.encoder(cnn_in)

        rnn_in = latent_var.view(batch_size, timesteps, -1)
        rnn_in = self.dropout(rnn_in)
        out = self.decoder(rnn_in)

        return out

class CNN_RNN(nn.Module):

    def __init__(self, 
                 n_classes, 
                 latent_size, 
                 n_cnn_layers, 
                 n_rnn_layers, 
                 n_rnn_hidden_dim,
                 channel_in=3, 
                 cnn_act_fn="relu", 
                 rnn_act_fn="relu", 
                 dropout_rate=0.1,
                 cnn_bn=False, 
                 bidirectional=False):
        
        super(CNN_RNN, self).__init__()

        self.encoder = CNN_Encoder(latent_size, n_cnn_layers, intermediate_act_fn=cnn_act_fn, use_batchnorm=cnn_bn, channel_in=channel_in)
        self.decoder = RNN_Decoder(latent_size, n_rnn_hidden_dim, n_classes, n_rnn_layers, intermediate_act_fn=rnn_act_fn, bidirectional=bidirectional)

        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):

        batch_size, timesteps, C, H, W = x.size()

        cnn_in = x.view(batch_size * timesteps, C, H, W)
        latent_var = self.encoder(cnn_in)

        rnn_in = latent_var.view(batch_size, timesteps, -1)
        rnn_in = self.dropout(rnn_in)
        out = self.decoder(rnn_in)

        return out

class ResNet_LSTM(nn.Module):

    def __init__(self, 
                 n_classes, 
                 latent_size, 
                 n_rnn_layers, 
                 channel_in=3, 
                 rnn_act_fn="relu", 
                 bidirectional=False, 
                 resnet_opt="resnet18"):
        
        super(ResNet_LSTM, self).__init__()

        if resnet_opt == "resnet18":
            self.CNN = resnet18(pretrained=True)
        elif resnet_opt == "resnet101":
            self.CNN = resnet101(pretrained=True)
        
        self.CNN.fc = nn.Sequential(nn.Linear(self.CNN.fc.in_features, latent_size))

        self.LSTM = LSTM_Decoder(latent_size, 3, n_classes, n_rnn_layers, intermediate_act_fn=rnn_act_fn, bidirectional=bidirectional)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()

        cnn_in = x.view(batch_size * timesteps, C, H, W)
        latent_var = self.CNN(cnn_in)

        rnn_in = latent_var.view(batch_size, timesteps, -1)
        out = self.RNN(rnn_in)

        out = self.softmax(out)
        return out