import torch
import torch.nn as nn
import torch.nn.functional as F


class NonLocalLayer(nn.Module):
    def __init__(self, dim_in, dim_out, dim_inter, norm_layer):
        super(NonLocalLayer, self).__init__()
        # TODO: Third Version, Need check

        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=dim_in, out_channels=dim_inter,
                kernel_size=1, stride=1, padding=0),
            norm_layer(dim_inter),
        )
        self.f_query = self.f_key
        self.f_value = nn.Conv2d(in_channels=dim_in, out_channels=dim_inter,
                                 kernel_size=1, stride=1, padding=0)

        self.W = nn.Conv2d(in_channels=dim_inter, out_channels=dim_out,
            kernel_size=1, stride=1, padding=0)
        nn.init.constant(self.W.weight, 0)
        nn.init.constant(self.W.bias, 0)

    def forward(self, data_input):

        value = self.f_value(data_input)
        B, C, H, W = value.size()
        value = value.view(B, C, -1)
        value = value.permute(0, 2, 1)
        query = self.f_query(data_input).view(B, C, -1)
        query = query.permute(0, 2, 1)
        key = self.f_key(data_input).view(B, C, -1)

        sim_map = torch.matmul(query, key)
        sim_map = (C**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(B, C, H, W)
        context = self.W(context)

        return context


class EmbededNonLocalLayer(nn.Module):
    def __init__(self, dim_in, dim_out, dim_inter, norm_layer):
        super(EmbededNonLocalLayer, self).__init__()

        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=dim_in, out_channels=dim_inter,
                kernel_size=1, stride=1, padding=0),
            norm_layer(dim_inter),
        )
        self.f_query = self.f_key
        self.f_value = nn.Conv2d(in_channels=dim_in, out_channels=dim_inter,
                                 kernel_size=1, stride=1, padding=0)

        self.W = nn.Conv2d(in_channels=dim_inter, out_channels=dim_out,
            kernel_size=1, stride=1, padding=0)
        nn.init.constant(self.W.weight, 0)
        nn.init.constant(self.W.bias, 0)

        # EmbedSA -------------------------------------
        self.K = 9
        self.f_value_2 = nn.AdaptiveAvgPool2d(self.K)
        self.f_value_2_conv = nn.Conv2d(
            in_channels=dim_inter, out_channels=dim_inter, kernel_size=1, padding=0, stride=1
        )


    def forward(self, data_input):

        value = self.f_value(data_input)

        B, C, H, W = value.size()
        value_2 = self.f_value_2(value)
        value_2 = self.f_value_2_conv(value_2) # B, C, K, K

        value = value.view(B, C, -1)
        value = value.permute(0, 2, 1) # B, N, C

        value_2 = value_2.view(B, C, -1)

        query = self.f_query(data_input).view(B, C, -1)
        query = query.permute(0, 2, 1)
        key = self.f_key(data_input).view(B, C, -1)

        sim_map = torch.matmul(query, key)
        sim_map = (C ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        sim_map_value = torch.matmul(value, value_2) # B, N, K
        sim_map_value = (C ** -0.5) * sim_map_value
        sim_map_value = F.softmax(sim_map_value, dim=-1)

        sim_map_new = torch.matmul(sim_map, sim_map_value) # B, N, K

        context = torch.matmul(sim_map_new, value_2.permute(0, 2, 1)) # B, N, C
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(B, C, H, W)
        context = self.W(context)

        return context


class EmNonLocalLayer(nn.Module):
    def __init__(self, dim_in, dim_out, dim_inter, norm_layer):
        super(EmNonLocalLayer, self).__init__()
        # TODO: Third Version, Need check

        self.f_query = nn.Sequential(
            nn.Conv2d(in_channels=dim_in, out_channels=dim_inter,
                kernel_size=1, stride=1, padding=0),
            norm_layer(dim_inter),
        )
        self.f_value = nn.Conv2d(in_channels=dim_in, out_channels=dim_inter,
                                 kernel_size=1, stride=1, padding=0)

        self.W = nn.Conv2d(in_channels=dim_inter, out_channels=dim_out,
            kernel_size=1, stride=1, padding=0)
        nn.init.constant(self.W.weight, 0)
        nn.init.constant(self.W.bias, 0)
        self.f_value_2 = nn.AdaptiveAvgPool2d(9)
        self.f_value_conv = nn.Conv2d(dim_inter, dim_inter, kernel_size=1, padding=0, stride=1)

    def forward(self, data_input):
        # Value attention
        value = self.f_value(data_input)
        B, C, H, W = value.size()
        value_hat = self.f_value_2(value)
        value_hat = self.f_value_conv(value_hat) # B C K* K
        value_hat = value_hat.view(B, C, -1) # B, C, K*K

        value = value.view(B, C, -1)

        # Bases generation
        sim_bases = torch.matmul(value_hat.permute(0, 2, 1), value) # B, K*K, N
        sim_bases = (C**-0.5) * sim_bases
        sim_bases = F.softmax(sim_bases, dim=-1)  # B K*K, N

        bases = torch.matmul(sim_bases, value.permute(0, 2, 1)) # B K*K C
        bases = bases.permute(0, 2, 1).contiguous() + value_hat  # B, C, K*K

        # Query attention
        query = self.f_query(data_input).view(B, C, -1)
        query = query.permute(0, 2, 1)

        sim_map = torch.matmul(query, bases)
        sim_map = (C**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1) # B N K*K

        # Feature generation
        context = torch.matmul(sim_map, bases.permute(0, 2, 1)) # B, N, C
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(B, C, H, W)
        context = self.W(context)

        return context



if __name__ == '__main__':
    from layers.batch_norm import NaiveSyncBatchNorm
    m = EmNonLocalLayer(dim_in=512, dim_inter=256, dim_out=512, norm_layer=NaiveSyncBatchNorm).cuda()
    x = torch.randn(2, 512, 64, 64).cuda()

    import time
    st = time.time()
    for i in range(100):
        y = m(x)
    end = time.time()

    print(" 100 iter using %f" % (end - st))
    print(y.size())
