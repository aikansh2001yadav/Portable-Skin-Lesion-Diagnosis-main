from torch import nn
from metablock import MetaBlock
from metanet import MetaNet
import torch
import warnings
from torchvision import models

class MyMobilenet (nn.Module):

    def __init__(self, mobilenet, num_class, neurons_reducer_block=256, freeze_conv=False, p_dropout=0.5,
                 comb_method=None, comb_config=None, n_feat_conv=2048):

        super(MyMobilenet, self).__init__()

        _n_meta_data = 0
        if comb_method is not None:
            if comb_config is None:
                raise Exception("You must define the comb_config since you have comb_method not None")

            if comb_method == 'metablock':
                if isinstance(comb_config, int):
                    self.comb_feat_maps = 40
                    self.comb = MetaBlock(self.comb_feat_maps, comb_config)
                elif isinstance(comb_config, list):
                    self.comb = MetaBlock(self.comb_feat_maps, comb_config[1])
                    self.comb_feat_maps = comb_config[0]
                else:
                    raise Exception(
                        "comb_config must be a list or int to define the number of feat maps and the metadata")
            elif comb_method == 'concat':
                if not isinstance(comb_config, int):
                    raise Exception("comb_config must be int for 'concat' method")
                _n_meta_data = comb_config
                self.comb = 'concat'
            elif comb_method == 'metanet':
                if isinstance(comb_config, int):
                    self.comb_feat_maps = 20
                    self.comb = MetaNet(comb_config, 40, 32)
                elif isinstance(comb_config, list):
                    self.comb = MetaNet(comb_config[0], comb_config[1], comb_config[2])
                    self.comb_feat_maps = comb_config[2]
                else:
                    raise Exception(
                        "comb_config must be a list or int to define the number of feat maps and the metadata")
            else:
                raise Exception("There is no comb_method called " + comb_method + ". Please, check this out.")
        else:
            self.comb = None

        self.features = nn.Sequential(*list(mobilenet.children())[:-1])
        self.conv1x1 = nn.Conv2d(1280, 2048, 1)

        # freezing the convolution layers
        if freeze_conv:
            for param in self.features.parameters():
                param.requires_grad = False

        # Feature reducer
        if neurons_reducer_block > 0:
            self.reducer_block = nn.Sequential(
                nn.Linear(n_feat_conv, neurons_reducer_block),
                nn.BatchNorm1d(neurons_reducer_block),
                nn.ReLU(),
                nn.Dropout(p=p_dropout)
            )
        else:
            if comb_method == 'concat':
                warnings.warn("You're using concat with neurons_reducer_block=0. Make sure you're doing it right!")
            self.reducer_block = None

        # Here comes the extra information (if applicable)
        if neurons_reducer_block > 0:
            self.classifier = nn.Linear(neurons_reducer_block + _n_meta_data, num_class)
        else:
            self.classifier = nn.Linear(n_feat_conv + _n_meta_data, num_class)

    def forward(self, img, meta_data=None):

        # Checking if when passing the metadata, the combination method is set
        if meta_data is not None and self.comb is None:
            raise Exception("There is no combination method defined but you passed the metadata to the model!")
        if meta_data is None and self.comb is not None:
            raise Exception("You must pass meta_data since you're using a combination method!")
      
        x = self.features(img)
        x = self.conv1x1(x)
        x = x.mean([2, 3])

        if self.comb == None:
            x = x.view(x.size(0), -1) # flatting
            if self.reducer_block is not None:
                x = self.reducer_block(x)  # feat reducer block
        elif self.comb == 'concat':
            x = x.view(x.size(0), -1) # flatting
            if self.reducer_block is not None:
                x = self.reducer_block(x) # feat reducer block. In this case, it must be defined
            x = torch.cat([x, meta_data], dim=1) # concatenation
        elif isinstance(self.comb, MetaBlock):
            x = x.view(x.size(0), self.comb_feat_maps, 32, -1).squeeze(-1) # getting the feature maps
            x = self.comb(x, meta_data.float()) # applying MetaBlock
            x = x.view(x.size(0), -1) # flatting
            if self.reducer_block is not None:
                x = self.reducer_block(x)  # feat reducer block
        elif isinstance(self.comb, MetaNet):
            x = x.view(x.size(0), self.comb_feat_maps, 8, 8).squeeze(-1)  # getting the feature maps
            x = self.comb(x, meta_data.float())  # applying metanet
            x = x.view(x.size(0), -1)  # flatting
            if self.reducer_block is not None:
                x = self.reducer_block(x)  # feat reducer block
        x = self.classifier(x)
        return x


def extract_feats_mobilenet(model, data, student=True):
    if not student:
        model.eval()
    
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            if not student:
                activation[name] = output.detach()
            else:
                activation[name] = output
        return hook

    def get_activation_pre(name):
        def hook(model, input, output):
            if not student:
                activation[name] = input[0].detach()
            else:
                activation[name] = input[0]
        return hook

    # modules = model.named_children()   

    # for name, module in modules:  
    #     if name == 'features':
    #         submod_ = module.named_children()  
    #         for name_sub_, module_sub_ in submod_:
    #             submod = module_sub_.named_children()
                # for name_sub, module_sub in submod:
                    # if name_sub == '18':
                        # module_sub.register_forward_hook(get_activation('feats_7x7'))  

    model.classifier.register_forward_hook(get_activation_pre('avg_pool'))  
    model.classifier.register_forward_hook(get_activation('logits'))    
    model.conv1x1.register_forward_hook(get_activation('feats_7x7'))
    output = model(data)

    return activation