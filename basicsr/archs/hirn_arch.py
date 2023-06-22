import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import warnings
import math

from basicsr.archs.arch_util import flow_warp
from basicsr.archs.basicvsr_arch import ConvResidualBlocks
from basicsr.archs.spynet_arch import SpyNet
from basicsr.ops.dcn import ModulatedDeformConvPack
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class HiRN(nn.Module):
    """HiRN (Hierarchical Recurrent Neural Network) Structure.

    Support either x4 upsampling or same size output. Since DCN is used in this
    model, it can only be used with CUDA enabled. If CUDA is not enabled,
    feature alignment will be skipped. Besides, we adopt the official DCN
    implementation and the version of torch need to be higher than 1.9.

    Args:
        mid_channels (int, optional): Channel number of the intermediate
            features. Default: 64.
        num_blocks (int, optional): The number of residual blocks in each
            propagation branch. Default: 7.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.
        is_low_res_input (bool, optional): Whether the input is low-resolution
            or not. If False, the output resolution is equal to the input
            resolution. Default: True.
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
        cpu_cache_length (int, optional): When the length of sequence is larger
            than this value, the intermediate features are sent to CPU. This
            saves GPU memory, but slows down the inference speed. You can
            increase this number if you have a GPU with large memory.
            Default: 100.
    """

    def __init__(self,
                 in_channels=3,
                 mid_channels=64,
                 num_blocks=7,
                 max_residue_magnitude=10,
                 is_low_res_input=True,
                 spynet_path=None,
                 cpu_cache_length=100):

        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.is_low_res_input = is_low_res_input
        self.cpu_cache_length = cpu_cache_length

        # optical flow
        self.spynet = SpyNet(spynet_path)

        # feature extraction module
        if is_low_res_input:
            self.feat_extract = ConvResidualBlocks(3, mid_channels, 5)
        else:
            self.feat_extract = nn.Sequential(
                nn.Conv2d(3, mid_channels, 3, 2, 1), nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(mid_channels, mid_channels, 3, 2, 1), nn.LeakyReLU(negative_slope=0.1, inplace=True),
                ConvResidualBlocks(mid_channels, mid_channels, 5))

        # propagation branches
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        self.twa = nn.ModuleDict()
        modules = ['backward_1', 'forward_1', 'group_1']
        for i, module in enumerate(modules):
            if torch.cuda.is_available():
                self.deform_align[module] = SecondOrderDeformableAlignment(
                    2 * mid_channels,
                    mid_channels,
                    3,
                    padding=1,
                    deformable_groups=16,
                    max_residue_magnitude=max_residue_magnitude)
            self.twa[module] = nn.Sequential(
                TemporalAttention(),
                WaveletAttention())
            self.backbone[module] = ConvResidualBlocks((4 + 2 + i) * mid_channels, mid_channels, num_blocks)

        # upsampling module
        self.reconstruction = ConvResidualBlocks(4 * mid_channels, mid_channels, 5)
        self.upconv1 = nn.Conv2d(mid_channels, mid_channels * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(mid_channels, 64 * 4, 3, 1, 1, bias=True)

        self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # check if the sequence is augmented by flipping
        self.is_mirror_extended = False

        if len(self.deform_align) > 0:
            self.is_with_alignment = True
        else:
            self.is_with_alignment = False
            warnings.warn('Deformable alignment module is not added. '
                          'Probably your CUDA is not configured correctly. DCN can only '
                          'be used with CUDA enabled. Alignment is skipped now.')

    def check_if_mirror_extended(self, lqs):
        """Check whether the input is a mirror-extended sequence.

        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the (t-1-i)-th frame.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with shape (n, t, c, h, w).
        """

        if lqs.size(1) % 2 == 0:
            lqs_1, lqs_2 = torch.chunk(lqs, 2, dim=1)
            if torch.norm(lqs_1 - lqs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def compute_flow(self, lqs):
        """Compute optical flow using SPyNet for feature alignment.

        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the flows used for forward-time propagation \
                (current to previous). 'flows_backward' corresponds to the flows used for backward-time \
                propagation (current to next).
        """

        n, t, c, h, w = lqs.size()
        lqs_1 = lqs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)

        if self.is_mirror_extended:  # flows_forward = flows_backward.flip(1)
            flows_forward = flows_backward.flip(1)
        else:
            flows_forward = self.spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)

        if self.cpu_cache:
            flows_backward = flows_backward.cpu()
            flows_forward = flows_forward.cpu()

        return flows_forward, flows_backward

    def propagate(self, feats, flows_backward, flows_forward, module_name):
        """Propagate the latent features throughout the sequence.

        Args:
            feats dict(list[tensor]): Features from previous branches. Each
                component is a list of tensors with shape (n, c, h, w).
            flows_backward (tensor): Backward Optical flows with shape (n, t - 1, 2, h, w).
            flows_forward (tensor): Forward Optical flows with shape (n, t - 1, 2, h, w).
            module_name (str): The name of the propgation branches. Can either
                be 'backward', 'forward', 'group'.

        Return:
            dict(list[tensor]): A dictionary containing all the propagated \
                features. Each key in the dictionary corresponds to a \
                propagation branch, which is represented by a list of tensors.
        """

        n, t, _, h, w = flows_backward.size()

        frame_idx = range(0, t + 1)
        if t + 1 >= 16:
            group_l0_idx = [0, 0, 0, 0, 0, 0, 2, 4, 4, 6, 8, 8, 8, 10, 12, 12, 14]
            group_frame_idx = [0, 16, 8, 4, 2, 1, 3, 6, 5, 7, 12, 10, 9, 11, 14, 13, 15]
            group_l1_idx = [0, 16, 16, 8, 4, 2, 4, 8, 6, 8, 16, 12, 10, 12, 16, 14, 16]
            gop = 16
        else:
            group_l0_idx = [0, 0, 0, 0, 1, 3, 4]
            group_frame_idx = [0, 6, 3, 1, 2, 4, 5]
            group_l1_idx = [0, 6, 6, 3, 3, 6, 6]
            gop = 7
        flow_idx_b = frame_idx[::-1]
        flow_idx_f = range(-1, t)
        mapping_idx = list(range(0, len(feats['spatial'])))
        mapping_idx += mapping_idx[::-1]

        if 'backward' in module_name:
            frame_idx = frame_idx[::-1]

        feat_prop = flows_backward.new_zeros(n, self.mid_channels, h, w)

        if 'group' not in module_name:
            for i, idx in enumerate(frame_idx):
                cond_n1 = flows_backward.new_zeros(n, self.mid_channels, h, w)
                cond_n2 = flows_backward.new_zeros(n, self.mid_channels, h, w)
                feat_current = flows_backward.new_zeros(n, self.mid_channels, h, w)
                feat_f = flows_backward.new_zeros(n, self.mid_channels, h, w)
                if 'backward' in module_name:
                    feat_current = feats['spatial'][mapping_idx[idx]]
                    feat_f = feats['spatial'][mapping_idx[idx]]
                elif 'forward' in module_name:
                    feat_current = feats['spatial'][mapping_idx[idx]]
                    feat_f = feats['backward_1'][mapping_idx[idx]]
                if self.cpu_cache:
                    feat_current = feat_current.cuda()
                    feat_prop = feat_prop.cuda()
                # second-order deformable alignment
                if i > 0 and self.is_with_alignment:
                    if 'backward' in module_name:
                        flow_n1 = flows_backward[:, flow_idx_b[i], :, :, :]
                    elif 'forward' in module_name:
                        flow_n1 = flows_forward[:, flow_idx_f[i], :, :, :]
                    if self.cpu_cache:
                        flow_n1 = flow_n1.cuda()

                    cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))

                    # initialize second-order features
                    feat_n2 = torch.zeros_like(feat_prop)
                    flow_n2 = torch.zeros_like(flow_n1)
                    cond_n2 = torch.zeros_like(cond_n1)

                    if i > 1:  # second-order features
                        feat_n2 = feats[module_name][-2]
                        if self.cpu_cache:
                            feat_n2 = feat_n2.cuda()

                        if 'backward' in module_name:
                            flow_n2 = flows_backward[:, flow_idx_b[i - 1], :, :, :]
                        elif 'forward' in module_name:
                            flow_n2 = flows_forward[:, flow_idx_f[i - 1], :, :, :]
                        if self.cpu_cache:
                            flow_n2 = flow_n2.cuda()

                        flow_n2 = flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1))
                        cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))

                    # flow-guided deformable convolution
                    cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                    feat_prop = torch.cat([feat_prop, feat_n2], dim=1)
                    feat_prop = self.deform_align[module_name](feat_prop, cond, flow_n1, flow_n2)

                # temporal wavelet attention and backbone
                # backbone : residual blocks
                twa_input = torch.cat([cond_n1.unsqueeze(1), feat_f.unsqueeze(1), cond_n2.unsqueeze(1)], dim=1)
                feat_twa = self.twa[module_name](twa_input)
                feat = [feat_current] + [feats[k][idx] for k in feats if k not in ['spatial', module_name]] + [feat_prop] + [feat_twa]
                if self.cpu_cache:
                    feat = [f.cuda() for f in feat]

                feat = torch.cat(feat, dim=1)
                feat_prop = feat_prop + self.backbone[module_name](feat)
                feats[module_name].append(feat_prop)

                if self.cpu_cache:
                    feats[module_name][-1] = feats[module_name][-1].cpu()
                    torch.cuda.empty_cache()

            if 'backward' in module_name:
                feats[module_name] = feats[module_name][::-1]
        else:
            for num_f in range(t + 1):
                feats[module_name].append(torch.zeros_like(feats['forward_1'][0]))
            group_num = math.ceil((t + 1) / gop)
            for i in range(group_num):
                for l0_idx, f_idx, l1_idx in zip(group_l0_idx, group_frame_idx, group_l1_idx):
                    cond_l0 = flows_backward.new_zeros(n, self.mid_channels, h, w)
                    cond_l1 = flows_backward.new_zeros(n, self.mid_channels, h, w)
                    if i * gop + f_idx >= t + 1 or (i != 0 and f_idx == 0):
                        pass
                    else:
                        if i * gop + l0_idx >= t + 1:
                            l0_idx = f_idx
                        if i * gop + l1_idx >= t + 1:
                            l1_idx = t - i * gop

                        global_l0_idx = i * gop + l0_idx
                        global_f_idx = i * gop + f_idx
                        global_l1_idx = i * gop + l1_idx

                        feat_current = feats['spatial'][global_f_idx]
                        feat_f = feats['forward_1'][global_f_idx]
                        if self.cpu_cache:
                            feat_current = feat_current.cuda()

                        # l0 and l1 deformable alignment
                        # for group propagation, l0 replaces n1, l1 replaces n2
                        if global_f_idx > 0 and self.is_with_alignment:
                            # l0 features
                            feat_l0 = feats[module_name][global_l0_idx]
                            if self.cpu_cache:
                                feat_l0 = feat_l0.cuda()

                            flow_l0 = flows_forward[:, flow_idx_f[global_f_idx], :, :, :]
                            for inter_idx in range(global_f_idx - 1, global_l0_idx, -1):
                                flow_l0 = flow_l0 + flow_warp(flows_forward[:, flow_idx_f[inter_idx], :, :, :],
                                                              flow_l0.permute(0, 2, 3, 1))
                            if self.cpu_cache:
                                flow_l0 = flow_l0.cuda()

                            cond_l0 = flow_warp(feat_l0, flow_l0.permute(0, 2, 3, 1))

                            # l1 features
                            feat_l1 = feats[module_name][global_l1_idx]
                            if self.cpu_cache:
                                feat_l1 = feat_l1.cuda()

                            flow_l1 = torch.zeros_like(flow_l0)
                            if global_f_idx != global_l1_idx:
                                flow_l1 = flows_backward[:, global_f_idx, :, :, :]
                                if global_l1_idx > global_f_idx:
                                    for inter_idx in range(global_f_idx + 1, global_l1_idx, 1):
                                        flow_l1 = flow_l1 + flow_warp(flows_backward[:, flow_idx_f[inter_idx], :, :, :],
                                                                      flow_l1.permute(0, 2, 3, 1))
                            if self.cpu_cache:
                                flow_l1 = flow_l1.cuda()

                            cond_l1 = flow_warp(feat_l1, flow_l1.permute(0, 2, 3, 1))

                            # flow-guided deformable convolution
                            cond = torch.cat([cond_l0, feat_current, cond_l1], dim=1)
                            feat_prop = torch.cat([feat_l0, feat_l1], dim=1)
                            feat_prop = self.deform_align[module_name](feat_prop, cond, flow_l0, flow_l1)

                        # temporal wavelet attention and backbone
                        # backbone : residual blocks
                        twa_input = torch.cat([cond_l0.unsqueeze(1), feat_f.unsqueeze(1), cond_l1.unsqueeze(1)], dim=1)
                        feat_twa = self.twa[module_name](twa_input)
                        feat = [feat_current] + [feats[k][global_f_idx] for k in feats if k not in ['spatial', module_name]] + [feat_prop] + [feat_twa]
                        if self.cpu_cache:
                            feat = [f.cuda() for f in feat]

                        feat = torch.cat(feat, dim=1)
                        feat_prop = feat_prop + self.backbone[module_name](feat)
                        feats[module_name][global_f_idx] = feat_prop

                        if self.cpu_cache:
                            feats[module_name][-1] = feats[module_name][-1].cpu()
                            torch.cuda.empty_cache()

        return feats

    def upsample(self, lqs, feats):
        """Compute the output image given the features.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
            feats (dict): The features from the propagation branches.

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        outputs = []
        num_outputs = len(feats['spatial'])

        mapping_idx = list(range(0, num_outputs))
        mapping_idx += mapping_idx[::-1]

        for i in range(0, lqs.size(1)):
            hr = [feats[k].pop(0) for k in feats if k != 'spatial']
            hr.insert(0, feats['spatial'][mapping_idx[i]])
            hr = torch.cat(hr, dim=1)
            if self.cpu_cache:
                hr = hr.cuda()

            hr = self.reconstruction(hr)
            hr = self.lrelu(self.pixel_shuffle(self.upconv1(hr)))
            hr = self.lrelu(self.pixel_shuffle(self.upconv2(hr)))
            hr = self.lrelu(self.conv_hr(hr))
            hr = self.conv_last(hr)
            if self.is_low_res_input:
                hr += self.img_upsample(lqs[:, i, :, :, :])
            else:
                hr += lqs[:, i, :, :, :]

            if self.cpu_cache:
                hr = hr.cpu()
                torch.cuda.empty_cache()

            outputs.append(hr)

        return torch.stack(outputs, dim=1)

    def forward(self, lqs):
        """Forward function for GBR-WNN Plus.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        n, t, c, h, w = lqs.size()

        # whether to cache the features in CPU
        self.cpu_cache = True if t > self.cpu_cache_length else False

        if self.is_low_res_input:
            lqs_downsample = lqs.clone()
        else:
            lqs_downsample = F.interpolate(
                lqs.view(-1, c, h, w), scale_factor=0.25, mode='bicubic').view(n, t, c, h // 4, w // 4)

        # check whether the input is an extended sequence
        self.check_if_mirror_extended(lqs)

        feats = {}
        # compute images to features
        if self.cpu_cache:
            feats['spatial'] = []
            for i in range(0, t):
                feat = self.feat_extract(lqs[:, i, :, :, :]).cpu()
                feats['spatial'].append(feat)
                torch.cuda.empty_cache()
        else:
            feats_ = self.feat_extract(lqs.view(-1, c, h, w))
            h, w = feats_.shape[2:]
            feats_ = feats_.view(n, t, -1, h, w)
            feats['spatial'] = [feats_[:, i, :, :, :] for i in range(0, t)]

        # compute optical flow using the low-res inputs
        assert lqs_downsample.size(3) >= 64 and lqs_downsample.size(4) >= 64, (
            'The height and width of low-res inputs must be at least 64, '
            f'but got {h} and {w}.')
        flows_forward, flows_backward = self.compute_flow(lqs_downsample)

        # feature propgation
        for iter_ in [1]:
            for direction in ['backward', 'forward', 'group']:
                module = f'{direction}_{iter_}'

                feats[module] = []

                if flows_forward is None:
                    flows_forward = flows_backward.flip(1)

                feats = self.propagate(feats, flows_backward, flows_forward, module)

                if self.cpu_cache:
                    torch.cuda.empty_cache()

        return self.upsample(lqs, feats)


class SecondOrderDeformableAlignment(ModulatedDeformConvPack):
    """Second-order deformable alignment module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.
    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

        super(SecondOrderDeformableAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * self.out_channels + 4, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deformable_groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self):

        def _constant_init(module, val, bias=0):
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.constant_(module.weight, val)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, bias)

        _constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, extra_feat, flow_1, flow_2):
        extra_feat = torch.cat([extra_feat, flow_1, flow_2], dim=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset_1 = offset_1 + flow_1.flip(1).repeat(1, offset_1.size(1) // 2, 1, 1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1, offset_2.size(1) // 2, 1, 1)
        offset = torch.cat([offset_1, offset_2], dim=1)

        # mask
        mask = torch.sigmoid(mask)

        return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding,
                                             self.dilation, mask)

class TemporalAttention(nn.Module):
    ''' Temporal Attention
    '''

    def __init__(self, nf=64, nframes=3, center=1, input_nf=64):
        super().__init__()
        self.center = center
        self.nframes = nframes

        # temporal attention (before fusion conv)
        self.tAtt_1 = nn.Conv2d(input_nf, nf, 3, 1, 1, bias=True)
        self.tAtt_2 = nn.Conv2d(input_nf, nf, 3, 1, 1, bias=True)

        # final spatial attention
        self.sAtt_1 = nn.Conv2d(input_nf * nframes, nf, 3, 1, 1, bias=True)
        self.sAtt_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_4 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_5 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, aligned_fea):
        B, N, C, H, W = aligned_fea.size()  # N video frames
        #### temporal attention
        emb_ref = self.tAtt_2(aligned_fea[:, self.center, :, :, :].clone())
        emb = self.tAtt_1(aligned_fea.view(-1, C, H, W)).view(B, N, -1, H, W)  # [B, N, C(nf), H, W]

        cor_l = []
        for i in range(N):
            emb_nbr = emb[:, i, :, :, :]
            cor_tmp = torch.sum(emb_nbr * emb_ref, 1).unsqueeze(1)  # B, 1, H, W
            cor_l.append(cor_tmp)
        cor_prob = torch.sigmoid(torch.cat(cor_l, dim=1))  # B, N, H, W
        cor_prob = cor_prob.unsqueeze(2).repeat(1, 1, C, 1, 1)  # B, N, C, H, W
        aligned_fea = aligned_fea * cor_prob  # B, N, C, H, W

        #### fusion
        att = self.lrelu(self.sAtt_1(aligned_fea.view(B, -1, H, W)))  # B, nf (128), H, W
        att_add = att
        att = self.lrelu(self.sAtt_2(att))
        att = self.lrelu(self.sAtt_3(att))
        att = att + att_add  # B, nf (128), H, W
        att = self.lrelu(self.sAtt_4(att))
        att = self.lrelu(self.sAtt_5(att)) # [B, nf, H, W]

        return att

class WaveletAttention(nn.Module):
    ''' Wavelet Attention
    '''

    def __init__(self, nf=128):
        super().__init__()

        ### Discrete Wavelet Transform
        self.DWT = DWT()

    def forward(self, x_att):
        B, C, H, W = x_att.size()  # C means number of features
        x_att = x_att.unsqueeze(1)  # [B, 1, nf, H, W]
        dwt_x_att = self.DWT(x_att)  # [B, 1, 4, nf, H//2, W//2]
        dwt_x_att = dwt_x_att.squeeze(1)  # [B, 4, nf, H//2, W//2]
        x_att = x_att.squeeze(1)  # [B, nf, H, W]

        dwt_x_att_mul = []
        for i in range(4):
            up_dwt_x_att = F.interpolate(dwt_x_att[:, i, :, :, :], scale_factor=2, mode='bicubic', align_corners=False)
            up_dwt_x_att_sig = torch.sigmoid(up_dwt_x_att)
            up_dwt_x_att_mul = x_att * up_dwt_x_att_sig  # [B, nf, H, W]
            dwt_x_att_mul.append(up_dwt_x_att_mul)
        dwt_x_att_mul = torch.stack(dwt_x_att_mul, dim=1).view(B, -1, H, W)  # [B, 4, nf, H, W] -> [B, 4*nf, H, W]

        return dwt_x_att_mul

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return self.dwt(x)

    def dwt(self, x):

        x01 = x[:, :, :, 0::2, :] / 2
        x02 = x[:, :, :, 1::2, :] / 2
        x1 = x01[:, :, :, :, 0::2]
        x2 = x02[:, :, :, :, 0::2]
        x3 = x01[:, :, :, :, 1::2]
        x4 = x02[:, :, :, :, 1::2]
        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4

        return torch.cat((x_LL.unsqueeze(2), x_HL.unsqueeze(2), x_LH.unsqueeze(2), x_HH.unsqueeze(2)), 2)
