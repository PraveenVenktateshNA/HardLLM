import math

import torch
import torch.nn as nn

CLIPMIN = 1e-5


def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


class UniformAffineQuantizer(nn.Module):
    def __init__(
        self,
        n_bits: int = 8,
        symmetric: bool = False,
        per_channel_axes=[],
        metric="minmax",
        dynamic=False,
        dynamic_method="per_cluster",
        group_size=None,
        shape=None,
        use_learnable_step_size=False,
        lwc = False,
        **kwargs
    ):
        """
        support cluster quantize
        dynamic_method support per_token and per_cluster
        """
        super().__init__()
        self.symmetric = symmetric
        assert 2 <= n_bits <= 16, "bitwidth not supported"
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = 2 ** (n_bits) - 1
        self.per_channel_axes = per_channel_axes
        self.metric = metric
        self.cluster_counts = None
        self.cluster_dim = None

        self.scale = None
        self.zero_point = None
        self.round_zero_point = None

        self.cached_xmin = None
        self.cached_xmax = None
        self.dynamic = dynamic
        self.dynamic_method = dynamic_method

        self.deficiency = 0
        self.use_learnable_step_size = use_learnable_step_size

        if use_learnable_step_size:
            if group_size:
                dim1 = int(shape[0] * math.ceil(shape[1] / group_size))
                self.deficiency = shape[-1] % group_size
                if self.deficiency > 0:
                    self.deficiency = group_size - self.deficiency
                    assert self.symmetric  # support for mlc-llm quantization
            else:
                dim1 = shape[0]
        self.lwc = lwc
        init_value = 4.
        if lwc:
            if group_size:
                dim1 = int(shape[0]*math.ceil(shape[1]/group_size))
                self.deficiency = shape[-1]%group_size
                if self.deficiency > 0:
                    self.deficiency = group_size - self.deficiency
                    assert self.symmetric   # support for mlc-llm symmetric quantization
            else:
                dim1 = shape[0]
            self.upbound_factor = nn.Parameter(torch.ones((dim1,1))*init_value)
            self.lowbound_factor = nn.Parameter(torch.ones((dim1,1))*init_value)
            # self.upbound_factor = nn.Parameter(torch.ones((dim1, 1)) * (gamma if gamma is not None else init_value))
            # self.lowbound_factor = nn.Parameter(torch.ones((dim1, 1)) * (beta if beta is not None else init_value))

        self.sigmoid = nn.Sigmoid()

        self.enable = True
        self.group_size = group_size
        self.is_init = False
        


    def change_n_bits(self, n_bits):
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = 2 ** (n_bits) - 1

    def fake_quant(self, x, scale, round_zero_point):
        if self.deficiency > 0:
            pad_zeros = torch.zeros(
                (x.shape[0], self.deficiency), dtype=x.dtype, device=x.device
            )
            x = torch.cat((x, pad_zeros), dim=1)

        if self.group_size:
            assert len(x.shape) == 2, "only support linear layer now"
            dim1, dim2 = x.shape
            x = x.reshape(-1, self.group_size)
        x_int = round_ste(x / scale)
        if round_zero_point is not None:
            x_int = x_int.add(round_zero_point)
        x_int = x_int.clamp(self.qmin, self.qmax)
        x_dequant = x_int
        if round_zero_point is not None:
            x_dequant = x_dequant.sub(round_zero_point)
        x_dequant = x_dequant.mul(scale)
        if self.group_size:
            x_dequant = x_dequant.reshape(dim1, dim2)
        if self.deficiency > 0:
            x_dequant = x_dequant[:, : -self.deficiency]
        return x_dequant

    def forward(self, x: torch.Tensor):
        if self.n_bits >= 16 or not self.enable:
            return x
        if self.metric == "fix0to1":
            return x.mul_(2**self.n_bits - 1).round_().div_(2**self.n_bits - 1)

        if self.dynamic_method == "per_token" or self.dynamic_method == "per_channel":
            self.per_token_dynamic_calibration(x)
        else:
            raise NotImplementedError()

        x_dequant = self.fake_quant(
            x, self.scale.abs().clamp(min=CLIPMIN, max=1e4), self.round_zero_point
        )
        return x_dequant

    def per_token_dynamic_calibration(self, x):
        if self.group_size:
            if self.deficiency == 0:
                x = x.reshape(-1, self.group_size)
            else:
                pad_zeros = torch.zeros(
                    (x.shape[0], self.deficiency), dtype=x.dtype, device=x.device
                )
                x = torch.cat((x, pad_zeros), dim=1)
                x = x.reshape(-1, self.group_size)
        reduce_shape = [-1]
        xmin = x.amin(reduce_shape, keepdim=True)
        xmax = x.amax(reduce_shape, keepdim=True)
        if self.lwc:
            # xmax = self.sigmoid(self.upbound_factor)*xmax
            # xmin = self.sigmoid(self.lowbound_factor)*xmin
            # 提取 self.upbound_factor 的一个元素

            upbound_value = self.sigmoid(self.upbound_factor).view(-1)[0]  # 取出其中的标量值
            lowbound_value = self.sigmoid(self.lowbound_factor).view(-1)[0]
            if isinstance(upbound_value, torch.Tensor):
                if upbound_value.numel() == 1:  # 如果张量只有一个元素
                    upbound_value = upbound_value.item()  # 将张量转换为数值
                else:
                    upbound_value = upbound_value.flatten()[0].item()  # 提取第一个元素并转换为数值
            if isinstance(lowbound_value, torch.Tensor):
                if lowbound_value.numel() == 1:  # 如果张量只有一个元素
                    lowbound_value = lowbound_value.item()  # 将张量转换为数值
                else:
                    lowbound_value = lowbound_value.flatten()[0].item()  # 提取第一个元素并转换为数值
            # 使用该值生成与 xmax 形状相同的张量
            truncated_upbound_factor = torch.full_like(xmax, upbound_value)
            truncated_lowbound_factor = torch.full_like(xmin, lowbound_value)
            # 现在 truncated_upbound_factor 的形状与 xmax 一样
            xmax = truncated_upbound_factor * xmax
            xmin = truncated_lowbound_factor * xmin
            # xmax = self.sigmoid(self.upbound_factor).expand_as(xmax) * xmax
            # xmin = self.sigmoid(self.upbound_factor).expand_as(xmax) * xmax
        if self.symmetric:
            abs_max = torch.max(xmax.abs(), xmin.abs())
            scale = abs_max / (2 ** (self.n_bits - 1) - 1)
            # scale = scale.clamp(min=CLIPMIN, max=1e4)
            if self.use_learnable_step_size:
                if not self.is_init:
                    self.register_parameter("scale", torch.nn.Parameter(scale))
                    self.is_init = True
            else:
                self.scale = scale
            zero_point = (2 ** (self.n_bits - 1) - 1) * torch.ones_like(self.scale)
        else:
            range = xmax - xmin
            scale = range / (2**self.n_bits - 1)
            # self.scale = scale.clamp(min=CLIPMIN, max=1e4)
            if self.use_learnable_step_size:
                if not self.is_init:
                    del self.scale
                    self.register_parameter("scale", torch.nn.Parameter(scale))
                    self.is_init = True
            else:
                self.scale = scale
            zero_point = -(xmin) / (self.scale)
        self.round_zero_point = zero_point.clamp(min=-1e4, max=1e4).round()
