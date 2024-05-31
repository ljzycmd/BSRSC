""" ************************************************
* fileName: rscnet.py
* desc: Rolling shutter correction with adaptive warping
* author: mingdeng_cao
* date: 2021/09/17
* last revised: None
************************************************ """


import torch
import torch.nn as nn
import torch.nn.functional as F

from correlation_package.corr_cuda import Correlation

from simdeblur.model.build import BACKBONE_REGISTRY


class BasicBlock(nn.Module):
    def __init__(self, in_channels, act=nn.LeakyReLU(negative_slope=0.1), stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.act = act
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        identity = x

        out = self.act(self.conv1(x))
        out = self.conv2(out)

        out = out + identity
        out = self.act(out)

        return out


class Encoder(nn.Module):
    def __init__(self, in_channels, inner_channels, num_blocks=3) -> None:
        super().__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, 7, 1, 3),
            nn.LeakyReLU(0.1),
            *[BasicBlock(inner_channels) for i in range(num_blocks)]
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels*2, 3, 2, 1),
            nn.LeakyReLU(0.1),
            *[BasicBlock(inner_channels*2) for i in range(num_blocks)]
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(inner_channels*2, inner_channels*4, 3, 2, 1),
            nn.LeakyReLU(0.1),
            *[BasicBlock(inner_channels*4) for i in range(num_blocks)]
        )

    def forward(self, x, return_ms_feats=True):
        x0 = self.conv0(x)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)

        if return_ms_feats:
            return x2, x1, x0

        return x2


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=3, multi_image_predict=False):
        super().__init__()
        self.multi_image_predict = multi_image_predict

        self.conv2 = nn.Sequential(
            *[BasicBlock(in_channels) for i in range(num_blocks)],
        )
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, in_channels // 2, 3, 1, 1)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, 1, 1),
            nn.LeakyReLU(0.1),
            *[BasicBlock(in_channels // 2) for i in range(num_blocks)]
        )
        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, 1, 1)
        )

        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, 1, 1),
            nn.LeakyReLU(0.1),
            *[BasicBlock(in_channels // 4) for i in range(num_blocks)]
        )

        if multi_image_predict:
            self.img_predictor = nn.ModuleList([
                nn.Conv2d(in_channels // 4, out_channels, 3, 1, 1),
                nn.Conv2d(in_channels // 2, out_channels, 3, 1, 1),
                nn.Conv2d(in_channels, out_channels, 3, 1, 1)
            ])
        else:
            self.img_predictor = nn.Conv2d(
                in_channels // 4, out_channels, 3, 1, 1)

    def forward(self, x2, x1, x0):
        out_list = []
        # level 2
        out = self.conv2(x2)
        out_list.append(out)

        # level 1
        out = torch.cat([self.upsample2(out), x1], dim=1)
        out = self.conv1(out)
        out_list.append(out)

        # level 0
        out = torch.cat([self.upsample1(out), x0], dim=1)
        out = self.conv0(out)
        out_list.append(out)

        # reverse the output list
        out_list.reverse()
        img = []
        if self.multi_image_predict:
            for i in range(len(out_list)):
                img.append(self.img_predictor[i](out_list[i]))
        else:
            # predict corrected image
            img.append(self.img_predictor(out))

        return img


class AdaWarp(nn.Module):
    def __init__(self, in_channels=32, num_heads=1, dropout=0, num_flow=9, **kwargs):
        super().__init__()
        self.adamsa = nn.MultiheadAttention(in_channels, num_heads=num_heads, dropout=dropout)
        self.pos_encoding = nn.Parameter(torch.Tensor(num_flow, 1, 1))
        self.conv_ffn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.LeakyReLU(0.1)
        )

    def get_pos_encoding(self):
        return self.pos_encoding.unsqueeze(0).unsqueeze(-1)

    def feats_sampling(self, x, flow, interpolation='bilinear', padding_mode='zeros', align_corners=True):
        """
        Args:
            x: shape(B, C, H, W)
            flow: shape(B, H, W, 2)
        """
        if x.size()[-2:] != flow.size()[1:3]:
            raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
                             f'flow ({flow.size()[1:3]}) are not the same.')
        h, w = x.shape[-2:]
        # create mesh grid
        grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
        grid = torch.stack((grid_x, grid_y), 2).type_as(x)  # (w, h, 2)
        grid.requires_grad = False

        grid_flow = grid + flow
        # scale grid_flow to [-1,1]
        grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
        grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
        grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
        output = F.grid_sample(
            x,
            grid_flow,
            mode=interpolation,
            padding_mode=padding_mode,
            align_corners=align_corners)
        return output

    def forward(self, x, flows, masks, modulation=True, **kwargs):
        """
        Args:
            x: shape(B, C, H, W)
            x_ref: shape(B, C, H, W)
            flows: shape(B, 2, NF, H, W)
            masks: shape(B, NF, H, W)
        """
        B, C, H, W = x.shape
        residual = x
        if modulation:
            flows = flows * masks.unsqueeze(1)
        NF = flows.shape[2]

        sampled_feats = []
        for i in range(NF):
            warped = self.feats_sampling(x, flows[:, :, i].permute(0, 2, 3, 1))
            sampled_feats.append(warped)

        sampled_feats = torch.stack(sampled_feats, dim=1)  # (B, NF, C, H, W)

        sampled_feats += self.get_pos_encoding()
        q = x.permute(0, 2, 3, 1).flatten(0, 2).unsqueeze(0)  # (1, B*H*W, C)
        kv = sampled_feats.permute(1, 0, 3, 4, 2).flatten(1, 3)  # (NF, B*H*W, C)

        # attention to aggregate feature
        out = self.adamsa(query=q, key=kv, value=kv)[0]
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        out += residual
        # feed-forward network
        ffn_out = self.conv_ffn(out)
        ffn_out += out

        return ffn_out


class MotionNet(nn.Module):
    def __init__(self, in_channels=32, inner_channels=32, out_flow_channel=3, num_flow=1, max_displacement=4, base_flow_channel=2, base_feature_channel=32):
        super().__init__()
        self.corr = Correlation(
            pad_size=max_displacement,
            kernel_size=1,
            max_displacement=max_displacement,
            stride1=1,
            stride2=1,
            corr_multiply=1
        )
        self.act = nn.LeakyReLU(0.1)
        corr_dim = (2*max_displacement + 1) ** 2
        feature_dim = in_channels
        if base_feature_channel is not None:
            feature_dim = base_feature_channel
        if base_flow_channel is not None:
            feature_dim += base_flow_channel
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(corr_dim+feature_dim, inner_channels, 3, 1, 1),
            nn.LeakyReLU(0.1),
            BasicBlock(inner_channels),
        )
        self.generate_flows = nn.Conv2d(inner_channels, out_flow_channel*num_flow, 3, 1, 1)
        self.modulation = True if out_flow_channel > 2 else False

    def forward(self, x_ref, x_neighbor, base_flow=None, base_feature=None):
        x_ref = x_ref.contiguous()
        x_neighbor = x_neighbor.contiguous()
        corr = self.act(self.corr(x_ref, x_neighbor))

        B, C, H, W = x_ref.shape
        input_feats = F.interpolate(base_feature, scale_factor=2, mode="bilinear") if base_feature is not None else x_neighbor
        if base_flow is not None:
            # upsample the flows
            base_flow = F.interpolate(base_flow.flatten(1, 2), scale_factor=2, mode="bilinear") * 2
            input_feats = torch.cat([input_feats, base_flow], dim=1)
        feats = self.feature_extractor(torch.cat([corr, input_feats], dim=1))
        flows_and_mask = self.generate_flows(feats)  # (b, 2NF, H, W)
        if self.modulation:
            flow_x, flow_y, mask = torch.chunk(flows_and_mask, 3, dim=1)
            flows = torch.stack([flow_x, flow_y], dim=1)  # (b, 2, NF, H, W)
            # modulation the flows with predicted masks
            mask = torch.sigmoid(mask)  # (B, NF, H, W)
        else:
            flows = flows_and_mask
            mask = None
        return flows, mask, feats


class FlowGenerator(nn.Module):
    def __init__(self, in_channels=32, md=4, num_flow=2, refer_idx=None):
        super().__init__()
        self.ref_idx = refer_idx
        self.motion_net = nn.ModuleList([
            MotionNet(in_channels=in_channels*4, inner_channels=32, out_flow_channel=3, num_flow=num_flow, max_displacement=md, base_flow_channel=None, base_feature_channel=None),
            MotionNet(in_channels=in_channels*2, inner_channels=32, out_flow_channel=3, num_flow=num_flow, max_displacement=md, base_flow_channel=2*num_flow, base_feature_channel=32),
            MotionNet(in_channels=in_channels, inner_channels=32, out_flow_channel=3, num_flow=num_flow, max_displacement=md, base_flow_channel=2*num_flow, base_feature_channel=32)
        ])

    def forward(self, x2, x1, x0, **kwargs):
        """
        Args:
            x: (B, N, C, H, W)
        return:
            motion fiels used to warp features
        """
        N = x2.shape[1]
        self.ref_idx = N // 2 if self.ref_idx is None else self.ref_idx

        flows_list = []
        masks_list = []
        flows = None
        masks = None
        feats = None
        for i, x in enumerate([x2, x1, x0]):
            if N > 1:
                ref_feats = torch.cat([x[:, self.ref_idx], x[:, self.ref_idx+1], x[:, self.ref_idx]], dim=0)  # (BN, C, H, W)
            else:
                ref_feats = x.flatten(0, 1)  # (BN, C, H, W)
            ngr_feats = x.flatten(0, 1)  # neighboring features (BN, C, H, W)
            flows, masks, feats = self.motion_net[i](ref_feats, ngr_feats, base_flow=flows, base_feature=feats)
            BN, C, NF, H, W = flows.shape
            flows_temp = flows.reshape(BN//N, N, C, NF, H, W)

            flows_list.append(flows_temp)
            masks_list.append(masks.reshape(BN//N, N, NF, H, W))
        return flows_list, masks_list


class FusionNet(nn.Module):
    def __init__(self, in_channels=32, md=4, num_flow=2, num_frames=3, refer_idx=None, modulation=True, *args, **kwargs):
        super().__init__()
        self.num_frames = num_frames
        self.flow_generator = FlowGenerator(in_channels=in_channels, md=md, num_flow=num_flow, refer_idx=refer_idx)
        self.warper = nn.ModuleList([
                nn.ModuleList([AdaWarp(in_channels * 4, num_flow=num_flow, kernel_size=kwargs["kernel_size"]) for _ in range(num_frames)]),
                nn.ModuleList([AdaWarp(in_channels * 2, num_flow=num_flow, kernel_size=kwargs["kernel_size"]) for _ in range(num_frames)]),
                nn.ModuleList([AdaWarp(in_channels, num_flow=num_flow, kernel_size=kwargs["kernel_size"]) for _ in range(num_frames)])
        ])

        self.mf_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels * 4 * num_frames, in_channels * 4, 3, 1, 1),
                nn.LeakyReLU(0.1)
            ),
            nn.Sequential(
                nn.Conv2d(in_channels * 2 * num_frames, in_channels * 2, 3, 1, 1),
                nn.LeakyReLU(0.1)
            ),
            nn.Sequential(
                nn.Conv2d(in_channels * num_frames, in_channels, 3, 1, 1),
                nn.LeakyReLU(0.1)
            )
        ])

    def forward(self, x2, x1, x0, **kwargs):
        """
        Args:
            x: (B, N, C, H, W)
        return:
            features fused from multiple frames features.
        """
        x_out = []
        N = x2.shape[1]  # num frames
        (flows2, flows1, flows0), (masks2, masks1, masks0) = self.flow_generator(x2, x1, x0)
        for i, (x, flow, mask) in enumerate(zip([x2, x1, x0], [flows2, flows1, flows0], [masks2, masks1, masks0])):
            # warping
            B, N, C, H, W = x.shape
            time_map = kwargs.get("time_map")
            if time_map is not None:
                time_map = F.interpolate(time_map.flatten(0, 1), size=(H, W)).reshape(B, N, -1, H, W)  # (B, N, 1, H, W)
                flow = flow * time_map.unsqueeze(3)  # (B, N, 2, NF, H, W)
            warped = torch.stack([self.warper[i][j](x[:, j], flow[:, j], mask[:, j]) for j in range(self.num_frames)], dim=1) # (B, N, C, H, W)
            x_out.append(self.mf_fusion[i](warped.reshape(B, N*C, H, W).contiguous()))

        return x_out, (flows2.flatten(1, 3), flows1.flatten(1, 3), flows0.flatten(1, 3))


@BACKBONE_REGISTRY.register()
class AdaRSCNet(nn.Module):
    def __init__(self, in_channels, inner_channels, out_channels, num_flow=9, num_frames=3, warper="AdaWarp", kernel_size=None, multi_scale_loss=False):
        super().__init__()
        self.img_encoder = Encoder(in_channels, inner_channels)
        self.fusion_net = FusionNet(in_channels=inner_channels, md=4, num_flow=num_flow, num_frames=num_frames, warper=warper, kernel_size=kernel_size)
        self.img_decoder = Decoder(inner_channels * 4, out_channels, multi_image_predict=multi_scale_loss)

    def forward(self, x):
        """
        args:
            x: shape(B, N, C, H, W)
        return:
            corrected image: (B, 3, H, W)
        """
        B, N, _, H, W = x.shape
        time_map = x[:, :, -1:]  # (B, N, 1, H, W)
        x = x[:, :, :3]  # (B, N, 3, H, W)
        x2, x1, x0 = self.img_encoder(x.flatten(0, 1))  # (B*N, C, H, W)

        (x2, x1, x0), (flows2, flows1, flows0) = self.fusion_net(
            x2.reshape(B, N, -1, H//4, W//4), x1.reshape(B, N, -1, H//2, W//2), x0.reshape(B, N, -1, H, W), time_map=time_map)
        return (self.img_decoder(x2, x1, x0)), (flows2, flows1, flows0)


if __name__ == "__main__":
    pass
