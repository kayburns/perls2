import torch
import torch.nn as nn
from models.models_utils import init_weights, rescaleImage, filter_depth
from models.base_models.layers import CausalConv1D, Flatten, conv2d


class ProprioEncoder(nn.Module):
    def __init__(self, z_dim, initailize_weights=True):
        """
        Image encoder taken from selfsupervised code
        """
        super().__init__()
        self.z_dim = z_dim

        self.proprio_encoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 2 * self.z_dim),
            nn.LeakyReLU(0.1, inplace=True),
        )

        if initailize_weights:
            init_weights(self.modules())

    def forward(self, proprio):
        return self.proprio_encoder(proprio).unsqueeze(2)


class ForceEncoder(nn.Module):
    def __init__(self, z_dim, initailize_weights=True):
        """
        Force encoder taken from selfsupervised code
        """
        super().__init__()
        self.z_dim = z_dim

        self.frc_encoder = nn.Sequential(
            CausalConv1D(6, 16, kernel_size=2, stride=2),
            nn.LeakyReLU(0.1, inplace=True),
            CausalConv1D(16, 32, kernel_size=2, stride=2),
            nn.LeakyReLU(0.1, inplace=True),
            CausalConv1D(32, 64, kernel_size=2, stride=2),
            nn.LeakyReLU(0.1, inplace=True),
            CausalConv1D(64, 128, kernel_size=2, stride=2),
            nn.LeakyReLU(0.1, inplace=True),
            CausalConv1D(128, 2 * self.z_dim, kernel_size=2, stride=2),
            nn.LeakyReLU(0.1, inplace=True),
        )

        if initailize_weights:
            init_weights(self.modules())

    def forward(self, force):
        return self.frc_encoder(force)


class ImageEncoder(nn.Module):
    def __init__(self, z_dim, initailize_weights=True):
        """
        Image encoder taken from Making Sense of Vision and Touch
        """
        super().__init__()
        self.z_dim = z_dim

        self.img_conv1 = conv2d(3, 16, kernel_size=7, stride=2)
        self.img_conv2 = conv2d(16, 32, kernel_size=5, stride=2)
        self.img_conv3 = conv2d(32, 64, kernel_size=5, stride=2)
        self.img_conv4 = conv2d(64, 64, stride=2)
        self.img_conv5 = conv2d(64, 128, stride=2)
        self.img_conv6 = conv2d(128, self.z_dim, stride=2)
        self.img_encoder = nn.Linear(4 * self.z_dim, 2 * self.z_dim)
        self.flatten = Flatten()

        if initailize_weights:
            init_weights(self.modules())

    def forward(self, image):
        # image encoding layers
        out_img_conv1 = self.img_conv1(image)
        #print (f"first_conv: {out_img_conv1.shape}")
        out_img_conv2 = self.img_conv2(out_img_conv1)
        #print (f"sec_conv: {out_img_conv2.shape}")
        out_img_conv3 = self.img_conv3(out_img_conv2)
        #print (f"third_conv: {out_img_conv3.shape}")
        out_img_conv4 = self.img_conv4(out_img_conv3)
        #print (f"fourth_conv: {out_img_conv4.shape}")
        out_img_conv5 = self.img_conv5(out_img_conv4)
        #print (f"fith_conv: {out_img_conv5.shape}")
        out_img_conv6 = self.img_conv6(out_img_conv5)
        print (f"sixth_conv: {out_img_conv6.shape}")

        img_out_convs = (
            out_img_conv1,
            out_img_conv2,
            out_img_conv3,
            out_img_conv4,
            out_img_conv5,
            out_img_conv6,
        )

        # image embedding parameters
        flattened = self.flatten(out_img_conv6)
        img_out = self.img_encoder(flattened).unsqueeze(2)

        return img_out, img_out_convs

class PerlsImageEncoder(nn.Module):
    def __init__(self, z_dim, initailize_weights=True):
        """
        Image encoder taken from Making Sense of Vision and Touch
        """
        super().__init__()
        self.z_dim = z_dim

        self.img_conv1 = conv2d(3, 16, kernel_size=7, stride=2)
        self.img_conv2 = conv2d(16, 32, kernel_size=5, stride=2)
        self.img_conv3 = conv2d(32, 64, kernel_size=5, stride=2)
        self.img_conv4 = conv2d(64, 64, stride=2)
        self.img_conv5 = conv2d(64, 128, stride=2)
        self.img_conv6 = conv2d(128, self.z_dim, stride=2)
        self.img_encoder = nn.Linear(16 * self.z_dim, 2 * self.z_dim)
        self.flatten = Flatten()

        if initailize_weights:
            init_weights(self.modules())

    def forward(self, image):
        # image encoding layers

        out_img_conv1 = self.img_conv1(image)
        #print (f"first_conv: {out_img_conv1.shape}")
        out_img_conv2 = self.img_conv2(out_img_conv1)
        #print (f"sec_conv: {out_img_conv2.shape}")
        out_img_conv3 = self.img_conv3(out_img_conv2)
        #print (f"third_conv: {out_img_conv3.shape}")
        out_img_conv4 = self.img_conv4(out_img_conv3)
        #print (f"fourth_conv: {out_img_conv4.shape}")
        out_img_conv5 = self.img_conv5(out_img_conv4)
        #print (f"fith_conv: {out_img_conv5.shape}")
        out_img_conv6 = self.img_conv6(out_img_conv5)
        #print (f"sixth_conv: {out_img_conv6.shape}")

        img_out_convs = (
            out_img_conv1,
            out_img_conv2,
            out_img_conv3,
            out_img_conv4,
            out_img_conv5,
            out_img_conv6,
        )

        # image embedding parameters
        flattened = self.flatten(out_img_conv6)
        img_out = self.img_encoder(flattened).unsqueeze(2)

        return img_out, img_out_convs

class DepthEncoder(nn.Module):
    def __init__(self, z_dim, initailize_weights=True):
        """
        Simplified Depth Encoder taken from Making Sense of Vision and Touch
        """
        super().__init__()
        self.z_dim = z_dim

        self.depth_conv1 = conv2d(1, 32, kernel_size=3, stride=2)
        self.depth_conv2 = conv2d(32, 64, kernel_size=3, stride=2)
        self.depth_conv3 = conv2d(64, 64, kernel_size=4, stride=2)
        self.depth_conv4 = conv2d(64, 64, stride=2)
        self.depth_conv5 = conv2d(64, 128, stride=2)
        self.depth_conv6 = conv2d(128, self.z_dim, stride=2)

        self.depth_encoder = nn.Linear(4 * self.z_dim, 2 * self.z_dim)
        self.flatten = Flatten()

        if initailize_weights:
            init_weights(self.modules())

    def forward(self, depth):
        # depth encoding layers
        out_depth_conv1 = self.depth_conv1(depth)
        out_depth_conv2 = self.depth_conv2(out_depth_conv1)
        out_depth_conv3 = self.depth_conv3(out_depth_conv2)
        out_depth_conv4 = self.depth_conv4(out_depth_conv3)
        out_depth_conv5 = self.depth_conv5(out_depth_conv4)
        out_depth_conv6 = self.depth_conv6(out_depth_conv5)

        depth_out_convs = (
            out_depth_conv1,
            out_depth_conv2,
            out_depth_conv3,
            out_depth_conv4,
            out_depth_conv5,
            out_depth_conv6,
        )

        # depth embedding parameters
        flattened = self.flatten(out_depth_conv6)
        depth_out = self.depth_encoder(flattened).unsqueeze(2)

        return depth_out, depth_out_convs

class PerlsDepthEncoder(nn.Module):
    def __init__(self, z_dim, initailize_weights=True):
        """
        Simplified Depth Encoder taken from Making Sense of Vision and Touch
        """
        super().__init__()
        self.z_dim = z_dim

        self.depth_conv1 = conv2d(1, 32, kernel_size=3, stride=2)
        self.depth_conv2 = conv2d(32, 64, kernel_size=3, stride=2)
        self.depth_conv3 = conv2d(64, 64, kernel_size=4, stride=2)
        self.depth_conv4 = conv2d(64, 64, stride=2)
        self.depth_conv5 = conv2d(64, 128, stride=2)
        self.depth_conv6 = conv2d(128, self.z_dim, stride=2)

        self.depth_encoder = nn.Linear(16 * self.z_dim, 2 * self.z_dim)
        self.flatten = Flatten()

        if initailize_weights:
            init_weights(self.modules())

    def forward(self, depth):
        # depth encoding layers
        out_depth_conv1 = self.depth_conv1(depth)
        out_depth_conv2 = self.depth_conv2(out_depth_conv1)
        out_depth_conv3 = self.depth_conv3(out_depth_conv2)
        out_depth_conv4 = self.depth_conv4(out_depth_conv3)
        out_depth_conv5 = self.depth_conv5(out_depth_conv4)
        out_depth_conv6 = self.depth_conv6(out_depth_conv5)

        depth_out_convs = (
            out_depth_conv1,
            out_depth_conv2,
            out_depth_conv3,
            out_depth_conv4,
            out_depth_conv5,
            out_depth_conv6,
        )

        # depth embedding parameters
        flattened = self.flatten(out_depth_conv6)
        depth_out = self.depth_encoder(flattened).unsqueeze(2)

        return depth_out, depth_out_convs

class TactoColorEncoder(nn.Module):
    def __init__(self, z_dim, initialize_weights=True):
        super().__init__()
        self.z_dim = z_dim

        self.conv_1 = conv2d(3, 16, kernel_size=7, stride=2)
        self.conv_2 = conv2d(16, 32, kernel_size=7, stride=2)
        self.conv_3 = conv2d(32, 32, kernel_size=7, stride=2)
        self.conv_4 = conv2d(32, 32, kernel_size=7, stride=2)
        self.flatten = Flatten()

        self.linear = nn.Linear(32 * 10 * 8, self.z_dim)
        if initialize_weights:
            init_weights(self.modules())
    
    def forward(self, tacto_image):
        out_conv_1 = self.conv_1(tacto_image)
        out_conv_2 = self.conv_2(out_conv_1)
        out_conv_3 = self.conv_3(out_conv_2)
        out_conv_4 = self.conv_4(out_conv_3)

        flat = self.flatten(out_conv_4)
        out = self.linear(flat)

        #print (f"out_conv_4_shape: {out.shape}")

        return out

class TactoDepthEncoder(nn.Module):
    def __init__(self, z_dim, initialize_weights=True):
        super().__init__()
        self.z_dim = z_dim

        self.conv_1 = conv2d(1, 16, kernel_size=7, stride=2)
        self.conv_2 = conv2d(16, 32, kernel_size=7, stride=2)
        self.conv_3 = conv2d(32, 32, kernel_size=7, stride=2)
        self.conv_4 = conv2d(32, 32, kernel_size=7, stride=2)
        self.flatten = Flatten()

        self.linear = nn.Linear(32 * 10 * 8, self.z_dim)
        if initialize_weights:
            init_weights(self.modules())
    
    
    def forward(self, tacto_depth):
        out_conv_1 = self.conv_1(tacto_depth)
        out_conv_2 = self.conv_2(out_conv_1)
        out_conv_3 = self.conv_3(out_conv_2)
        out_conv_4 = self.conv_4(out_conv_3)

        flat = self.flatten(out_conv_4)
        out = self.linear(flat)

        #print (f"out_conv_4_shape: {out.shape}")

        return out



class FullTactoColorEncoder(nn.Module):
    def __init__(self, z_dim, initialize_weights=True):
        super().__init__()
        self.z_dim = z_dim
        self.color_encoder = TactoColorEncoder(z_dim, initialize_weights)
        self.pre_process = rescaleImage

        self.linear = nn.Sequential(
            nn.Linear(2 * self.z_dim, self.z_dim), 
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, tacto_colors):
        tacto_left = tacto_colors[:, 0]
        tacto_left = self.pre_process(tacto_left)

        tacto_right = tacto_colors[:, 1]
        tacto_right = self.pre_process(tacto_right)

        out_left = self.color_encoder(tacto_left)
        out_right = self.color_encoder(tacto_right)

        conc_out = torch.cat([out_left, out_right], dim=1)

        out = self.linear(conc_out)
        return out        
        
class FullTactoDepthEncoder(nn.Module):
    def __init__(self, z_dim, initialize_weights=True):
        super().__init__()

        self.z_dim = z_dim
        self.depth_encoder = TactoDepthEncoder(z_dim, initialize_weights)
        self.pre_process = self.depth_preproc
        self.linear = nn.Sequential(
            nn.Linear(2 * self.z_dim, self.z_dim), 
            nn.LeakyReLU(0.1, inplace=True),
        )

    
    def depth_preproc(self, depth_img):
        depth_img = depth_img.transpose(1, 3).transpose(2, 3)
        depth_img = filter_depth(depth_img)

        return depth_img
    
    def forward(self, tacto_depth):
        tacto_left = tacto_depth[:, 0]
        tacto_left = self.pre_process(tacto_left)

        tacto_right = tacto_depth[:, 1]
        tacto_right = self.pre_process(tacto_right)

        out_left = self.depth_encoder(tacto_left)
        out_right = self.depth_encoder(tacto_right)

        conc_out = torch.cat([out_left, out_right], dim=1)
        out = self.linear(conc_out)

        return out

class TactoEncoder(nn.Module):
    def __init__(self, z_dim, initialize_weights=True):
        super().__init__()

        self.z_dim = z_dim
        self.full_depth_enc = FullTactoDepthEncoder(self.z_dim)
        self.full_color_enc = FullTactoColorEncoder(self.z_dim)

        self.linear_func = nn.Sequential(
            nn.Linear(2 * self.z_dim, self.z_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(self.z_dim, 2 * self.z_dim),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, tacto_colors, tacto_depth):
        color_enc = self.full_color_enc(tacto_colors)
        depth_enc = self.full_depth_enc(tacto_depth)

        conc_vec = torch.cat([color_enc, depth_enc], dim=1)
        lin_out = self.linear_func(conc_vec).unsqueeze(2)

        return lin_out


