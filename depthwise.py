import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
import yaml
# Define a custom depthwise separable convolutional block
class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# Create a custom U-Net model with depthwise separable convolutions
class CustomUNet(UNet2DConditionModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.apply_depthwise_separable_conv()

    def apply_depthwise_separable_conv(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                depthwise_separable_conv = DepthwiseSeparableConv2d(
                    module.in_channels,
                    module.out_channels,
                    module.kernel_size,
                    module.stride,
                    module.padding,
                    module.bias is not None,
                )
                module = depthwise_separable_conv


def convert_conv_to_depthwise_separable(conv_layer, bias=True):
    """
    Converts a regular convolutional layer to a depthwise separable convolutional layer.
    The weights and biases of the new layer are initialized with the weights and biases
    of the original convolutional layer.
    """
    in_channels = conv_layer.in_channels
    out_channels = conv_layer.out_channels
    kernel_size = conv_layer.kernel_size
    stride = conv_layer.stride
    padding = conv_layer.padding
    dilation = conv_layer.dilation

    depthwise_separable_conv = DepthwiseSeparableConv2d(
        in_channels, out_channels, kernel_size, stride, padding, bias
    )

    # Initialize the depthwise convolution weights
    depthwise_separable_conv.depthwise.weight.data = conv_layer.weight.data.clone()

    # Initialize the pointwise convolution weights
    pointwise_weight = torch.sum(conv_layer.weight.data, dim=1, keepdim=True)
    depthwise_separable_conv.pointwise.weight.data = pointwise_weight.clone()

    # Initialize the biases
    if bias:
        depthwise_separable_conv.depthwise.bias.data = conv_layer.bias.data.clone()
        depthwise_separable_conv.pointwise.bias.data = conv_layer.bias.data.clone()

    return depthwise_separable_conv

 reference_unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_base_model_path,
        subfolder="unet",
    ).to(dtype=weight_dtype, device="cuda")


    # Load the pre-trained model
    # pretrained_model = YourPretrainedModel(...)
    # pretrained_model.load_state_dict(torch.load("path/to/pretrained_weights.pth"))

    # Convert the convolutional layers to depthwise separable convolutions
    for module in pretrained_model.modules():
        if isinstance(module, nn.Conv2d):
            depthwise_separable_conv = convert_conv_to_depthwise_separable(module)
            module = depthwise_separable_conv



    # Load the YAML configuration file
    with open('./configs/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    v2 = False  # SD 2.1
    # Access the reference_unet_config based on args.v2
    if v2:
        unet_config = config['reference_unet_config']['v2']
        denoise_unet_config = config['denoising_unet_config']['v2']
    else:
        # SD 1.5
        unet_config = config['reference_unet_config']['default']
        denoise_unet_config = config['denoising_unet_config']['default']


    reference_unet = CustomUNet(**config["reference_unet_config"])
