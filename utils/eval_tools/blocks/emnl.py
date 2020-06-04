import torch
import torch.nn as nn
import torch.nn.functional as F

class AvgNonLocalLayer(nn.Module):
    def __init__(self, dim_in, dim_out, dim_inter, norm_layer):
        super(AvgNonLocalLayer, self).__init__()
        self.f_query = nn.Sequential(
            nn.Conv2d(in_channels=dim_in, out_channels=dim_inter,
                kernel_size=1, stride=1, padding=0),
            norm_layer(dim_inter),
            nn.ReLU()
        )
        self.f_value = self.f_query

        self.W = nn.Conv2d(in_channels=dim_inter, out_channels=dim_out,
            kernel_size=1, stride=1, padding=0)
        nn.init.constant(self.W.weight, 0)
        nn.init.constant(self.W.bias, 0)
        self.f_value_2 = nn.AdaptiveAvgPool2d(9)

    def forward(self, data_input):
        # Value attention
        value = self.f_value(data_input)
        B, C, H, W = value.size()
        value_hat = self.f_value_2(value)
        value_hat = value_hat.view(B, C, -1) # B, C, K*K

        # Query attention
        query = self.f_query(data_input).view(B, C, -1)
        query = query.permute(0, 2, 1)

        sim_map = torch.matmul(query, value_hat)
        sim_map = (C**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1) # B N K*K

        # Feature generation
        context = torch.matmul(sim_map, value_hat.permute(0, 2, 1)) # B, N, C
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(B, C, H, W)
        context = self.W(context)
        return context



class EmNonLocalLayer(nn.Module):
    def __init__(self, dim_in, dim_out, dim_inter, norm_layer):
        super(EmNonLocalLayer, self).__init__()
        self.f_query = nn.Sequential(
            nn.Conv2d(in_channels=dim_in, out_channels=dim_inter,
                kernel_size=1, stride=1, padding=0),
            norm_layer(dim_inter),
            nn.ReLU()
        )
        self.f_value = self.f_query

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

