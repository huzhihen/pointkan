import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('models')
from KANLinear import KANLinear


from pointnet2_ops import pointnet2_utils


def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    else:
        return nn.ReLU(inplace=True)


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


class GlobalLocalGrouper(nn.Module):
    def __init__(self, channel, groups, kneighbors, use_xyz=True, normalize="center", use_attention=False, **kwargs):
        """
        Give xyz[b,p,3] and fea[b,p,c], return new_xyz[b,g,3] and new_fea[b,g,k,c]
        :param groups: groups number
        :param kneighbors: k-nerighbors
        :param kwargs: others
        """
        super(GlobalLocalGrouper, self).__init__()
        self.groups = groups
        self.kneighbors = kneighbors
        self.use_xyz = use_xyz
        self.normalize = normalize.lower() if normalize else None
        self.use_attention = use_attention
        if self.normalize not in ["center", "anchor"]:
            print(f"Unrecognized normalize parameter (self.normalize), set to None. Should be one of [center, anchor].")
            self.normalize = None
        if self.normalize is not None:
            add_channel = 3 if self.use_xyz else 0
            self.affine_alpha = nn.Parameter(torch.ones([1, 1, 1, channel + add_channel]))
            self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, channel + add_channel]))

        if self.use_attention:
            attn_dim = channel + 3 if self.use_xyz else channel  # 相对坐标+距离占4维
            self.attn_mlp = nn.Sequential(
                nn.Linear(attn_dim, attn_dim // 2),
                nn.LeakyReLU(),
                nn.Linear(attn_dim // 2, 1)
            )

    def forward(self, xyz, points):
        B, N, C = xyz.shape
        G = self.groups
        xyz = xyz.contiguous()  # xyz [batch, points, xyz]
        # FPS采样中心点
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.groups).long()  # [B, npoint]
        new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3]
        new_points = index_points(points, fps_idx)  # [B, npoint, C]
        # K近邻搜索
        idx = knn_point(self.kneighbors, xyz, new_xyz)  # [B, npoint, k]
        # idx = query_ball_point(0.1, self.kneighbors, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)  # [B, npoint, K, 3]
        grouped_points = index_points(points, idx)  # [B, npoint, K, C]
        if self.use_xyz:
            grouped_points = torch.cat([grouped_points, grouped_xyz],dim=-1)  # [B, npoint, k, C+3]
        if self.normalize is not None:
            if self.normalize =="center":
                mean = torch.mean(grouped_points, dim=2, keepdim=True)  # [B, G, 1, C]
            if self.normalize =="anchor":
                mean = torch.cat([new_points, new_xyz], dim=-1) if self.use_xyz else new_points
                mean = mean.unsqueeze(dim=-2)  # [B, npoint, 1, C+3]
            std = torch.std((grouped_points-mean).reshape(B, -1), dim=-1, keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)  # [B, 1, 1, 1]
            grouped_points = (grouped_points-mean)/(std + 1e-5)  # [B, G, K, C]
            grouped_points = self.affine_alpha * grouped_points + self.affine_beta  # [B, G, K, C]

        if self.use_attention:
            center_feat = new_points.unsqueeze(2).expand(-1, -1, self.kneighbors, -1)  # [B, G, K, C]
            new_points = torch.cat([grouped_points, center_feat], dim=-1)  # [B, G, K, 2 * C]
            attn = torch.softmax(self.attn_mlp(grouped_points), dim=2)  # [B, G, K, 1]
            new_points = attn * new_points  # [B, G, K, 2 * C]
        else:
            new_points = torch.cat([grouped_points, new_points.view(B, G, 1, -1).repeat(1, 1, self.kneighbors, 1)], dim=-1)  # [B, G, K, 2 * C]
        return new_xyz, new_points


class Conv1dBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True, activation='relu'):
        super(Conv1dBNReLU, self).__init__()
        self.act = get_activation(activation)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act
        )

    def forward(self, x):
        return self.net(x)


class Conv1dBNReLURes(nn.Module):
    def __init__(self, channel, kernel_size=1, res_expansion=1.0, bias=True, activation='relu'):
        super(Conv1dBNReLURes, self).__init__()
        self.act = get_activation(activation)
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=int(channel * res_expansion),
                      kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(int(channel * res_expansion)),
            self.act
        )
        self.net2 = nn.Sequential(
            nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                      kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(channel)
        )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)


class MLPExtraction(nn.Module):
    def __init__(self, in_channels, out_channels, blocks=1, res_expansion=1, bias=True, activation='relu'):
        super(MLPExtraction, self).__init__()
        self.transfer = Conv1dBNReLU(in_channels, out_channels, bias=bias, activation=activation)
        operation = []
        for _ in range(blocks):
            operation.append(
                Conv1dBNReLURes(out_channels, res_expansion=res_expansion, bias=bias, activation=activation)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):
        x = self.transfer(x)
        x = self.operation(x)
        return x


class KANTransformer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(KANTransformer, self).__init__()
        self.kan = nn.Sequential(
            KANLinear(
                in_features=in_channels,
                out_features=out_channels,
                grid_size=5,
                spline_order=3,
                scale_noise=0.1,
                scale_base=1.0,
                scale_spline=1.0,
                base_activation=nn.SiLU,
                grid_eps=0.02,
                grid_range=[-1, 1],
            ),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x):
        return self.kan(x)


class KANLinearBNReLURes(nn.Module):
    def __init__(self, channel, res_expansion=1.0):
        super(KANLinearBNReLURes, self).__init__()
        self.net1 = nn.Sequential(
            KANLinear(
                in_features=channel,
                out_features=int(channel * res_expansion),
                grid_size=5,
                spline_order=3,
                scale_noise=0.1,
                scale_base=1.0,
                scale_spline=1.0,
                base_activation=nn.SiLU,
                grid_eps=0.02,
                grid_range=[-1, 1],
            ),
            nn.BatchNorm1d(int(channel * res_expansion)),
        )
        self.net2 = nn.Sequential(
            KANLinear(
                in_features=int(channel * res_expansion),
                out_features=channel,
                grid_size=5,
                spline_order=3,
                scale_noise=0.1,
                scale_base=1.0,
                scale_spline=1.0,
                base_activation=nn.SiLU,
                grid_eps=0.02,
                grid_range=[-1, 1],
            ),
            nn.BatchNorm1d(channel)
        )

    def forward(self, x):
        return self.net2(self.net1(x)) + x


class KANPreExtraction(nn.Module):
    def __init__(self, channels, out_channels, blocks=1, res_expansion=1, use_xyz=True):
        super(KANPreExtraction, self).__init__()
        in_channels = 3 + 2 * channels if use_xyz else 2 * channels
        self.kan_transformer = KANTransformer(in_channels, out_channels)
        kan_layers = []
        for _ in range(blocks):
            kan_layers.append(KANLinearBNReLURes(out_channels, res_expansion=res_expansion))
        self.kan_layers = nn.Sequential(*kan_layers)

    def forward(self, x):
        B, G, K, C = x.size()  # x: [B, G, K, C]
        x = x.reshape(B * G * K, C)  # x: [B * G * K, C]
        x = self.kan_transformer(x)  # x: [B * G * K, C]
        x = self.kan_layers(x)  # x: [B * G * K, C]
        _, C = x.size()
        x = x.reshape(B * G, K, C).permute(0, 2, 1)  # x: [B * G, C, K]
        x = F.adaptive_max_pool1d(x, 1)  # x: [B * G, C]
        x = x.reshape(B, G, -1).permute(0, 2, 1)  # x: [B, C, G]
        return x


class KANSubExtraction(nn.Module):
    def __init__(self, out_channels, blocks=1, res_expansion=1):
        super(KANSubExtraction, self).__init__()
        kan_layers = []
        for _ in range(blocks):
            kan_layers.append(KANLinearBNReLURes(out_channels, res_expansion=res_expansion))
        self.kan_layers = nn.Sequential(*kan_layers)

    def forward(self, x):
        B, C, G = x.size()  # x: [B, C, G]
        x = x.permute(0, 2, 1).reshape(B * G, C)  # x: [B * G, C]
        x = self.kan_layers(x)  # x: [B * G, C]
        x = x.reshape(B, G, -1).permute(0, 2, 1)  # x: [B, C, G]
        return x


class KANHead(nn.Module):
    def __init__(self, last_channels, class_channels):
        super(KANHead, self).__init__()
        self.kan_head = KANLinear(
            in_features=last_channels,
            out_features=class_channels,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            base_activation=nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
        )

    def forward(self, x):
        x = F.adaptive_max_pool1d(x, 1).squeeze(dim=-1)  # x: [B, C, G] -> x: [B, C]
        x = self.kan_head(x)  # x: [B, C] -> x: [B, self.class_num]
        return x


class Model(nn.Module):
    def __init__(self, points=1024, class_num=40, embed_dim=64, res_expansion=1.0,
                 activation="relu", bias=False, use_xyz=True, normalize="center",
                 dim_expansion=[2, 2, 2, 2], kan_pre_blocks=[1, 1, 1, 1], kan_sub_blocks=[1, 1, 1, 1],
                 k_neighbors=[32, 32, 32, 32], reducers=[2, 2, 2, 2], **kwargs):
        super(Model, self).__init__()
        self.points = points
        self.class_num = class_num
        self.embed_dim = embed_dim
        self.stages = len(kan_pre_blocks)
        assert len(kan_pre_blocks) == len(kan_sub_blocks) == len(k_neighbors) == len(reducers) == len(dim_expansion), \
            "Please check stage number consistent for pre_blocks, pos_blocks k_neighbors, reducers."
        self.global_local_grouper_list = nn.ModuleList()
        self.kan_pre_blocks_list = nn.ModuleList()
        self.kan_sub_blocks_list = nn.ModuleList()
        last_channel = self.embed_dim
        anchor_points = self.points

        # KAN Network
        self.embedding = MLPExtraction(3, embed_dim, bias=bias, activation=activation)
        for i in range(len(kan_pre_blocks)):
            out_channel = last_channel * dim_expansion[i]
            kan_pre_block_num = kan_pre_blocks[i]
            kan_sub_block_num = kan_sub_blocks[i]
            kneighbor = k_neighbors[i]
            reduce = reducers[i]
            anchor_points = anchor_points // reduce
            # append local_grouper_list
            global_local_grouper = GlobalLocalGrouper(last_channel, anchor_points, kneighbor, use_xyz, normalize)
            self.global_local_grouper_list.append(global_local_grouper)

            # append kan_pre_block_list
            kan_pre_block_module = KANPreExtraction(last_channel, out_channel, kan_pre_block_num,
                                                    res_expansion=res_expansion, use_xyz=use_xyz)
            self.kan_pre_blocks_list.append(kan_pre_block_module)

            # append kan_sub_block_list
            kan_sub_block_module = KANSubExtraction(out_channel, kan_sub_block_num,
                                                    res_expansion=res_expansion)
            self.kan_sub_blocks_list.append(kan_sub_block_module)

            last_channel = out_channel

        self.kan_head = KANHead(last_channels=last_channel, class_channels=self.class_num)

    def forward(self, x):
        xyz = x.permute(0, 2, 1)  # x: [B, 3, N], xyz: [B, N, 3]
        x = self.embedding(x)  # x: [B, C, N]
        for i in range(self.stages):
            # Give xyz[B, N, 3] and feature[B, C, N], return new_xyz[B, G, 3] and new_fea[B, G, K, C * 2]
            xyz, x = self.global_local_grouper_list[i](xyz, x.permute(0, 2, 1))  # xyz: [B, G, 3]  x: [B, G, K, C * 2]
            x = self.kan_pre_blocks_list[i](x)  # x: [B, C * 2, G]
            x = self.kan_sub_blocks_list[i](x)  # x: [B, C * 2, G]

        x = self.kan_head(x)
        return x


def pointKAN(num_classes=40, **kwargs) -> Model:
    return Model(points=1024, class_num=num_classes, embed_dim=64, res_expansion=1.0,
                 activation="relu", bias=False, use_xyz=True, normalize="anchor",
                 dim_expansion=[2, 2], kan_pre_blocks=[1, 1], kan_sub_blocks=[1, 1],
                 k_neighbors=[32, 32], reducers=[2, 2], **kwargs)
