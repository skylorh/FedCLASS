import copy
from torch import nn
from convs.resnet_client import resnet8, resnet16, resnet20, resnet32, resnet44, resnet56, resnet110
from convs.linears import SimpleLinear

def get_convnet(convnet_type, pretrained=False, path=None):
    name = convnet_type.lower()
    if name == "resnet8":
        return resnet8(pretrained=pretrained, path=path)
    elif name == "resnet16":
        return resnet16(pretrained=pretrained, path=path)
    elif name == "resnet20":
        return resnet20(pretrained=pretrained, path=path)
    elif name == "resnet32":
        return resnet32(pretrained=pretrained, path=path)
    elif name == "resnet44":
        return resnet44(pretrained=pretrained, path=path)
    elif name == "resnet56":
        return resnet56(pretrained=pretrained, path=path)
    elif name == "resnet110":
        return resnet110(pretrained=pretrained, path=path)
    else:
        raise NotImplementedError("Unknown type {}".format(convnet_type))

class BaseNet(nn.Module):
    def __init__(self, convnet_type, pretrained):
        super(BaseNet, self).__init__()

        self.convnet = get_convnet(convnet_type, pretrained)
        self.fc = None

    def update_fc(self, len):
        fc = self.generate_fc(self.feature_dim, len)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim, bias=True)

        return fc
    
    def init_fc(self, len):
        self.fc = self.generate_fc(self.feature_dim, len)

    @property
    def feature_dim(self):
        return self.convnet.out_dim

    def forward(self, x):
        x = self.convnet(x)
        logits = self.fc(x)

        return logits

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self

class ICARLNet(BaseNet):
    def __init__(self, convnet_type, pretrained):
        super().__init__(convnet_type, pretrained)
    
class OURSNet(BaseNet):
    def __init__(self, convnet_type, pretrained):
        super().__init__(convnet_type, pretrained)
        
class GLFCNet(BaseNet):
    def __init__(self, convnet_type, pretrained):
        super().__init__(convnet_type, pretrained)

class CENet(BaseNet):
    def __init__(self, convnet_type, pretrained):
        super().__init__(convnet_type, pretrained)
        
class WANet(BaseNet):
    def __init__(self, convnet_type, pretrained):
        super().__init__(convnet_type, pretrained)

