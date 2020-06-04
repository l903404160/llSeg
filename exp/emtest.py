import torch
import torch.nn.functional as F

def _l2norm(inp, dim):
    '''Normlize the inp tensor with l2-norm.
    Returns a tensor where each sub-tensor of input along the given dim is
    normalized such that the 2-norm of the sub-tensor is equal to 1.
    Arguments:
        inp (tensor): The input tensor.
        dim (int): The dimension to slice over to get the ssub-tensors.
    Returns:
        (tensor) The normalized tensor.
    '''
    return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))

# simulate the input feature
# input_fea = torch.zeros(64, 100, 100)
# input_fea[:32, 10:20, 10:20] = 0.3
# input_fea[32:, 10:20, 10:20] = 0.7
#
# input_fea[:16, 20:30, 20:30] = 0.2
# input_fea[16:, 20:30, 20:30] = 0.4

input_fea = torch.randn(64, 100, 100)


c,h,w = input_fea.size()

# k bases
mu = torch.zeros(64, 64)
for i in range(64):
    mu[:, i] = torch.randn(64)

# e
input_fea = input_fea.view(c, h*w).unsqueeze(0)
input_fea_T = input_fea.permute(0, 2, 1)
mu = mu.unsqueeze(0)

for i in range(6):
    z = torch.bmm(input_fea_T, mu)
    temp = z.numpy()
    z = F.softmax(z, dim=1)
    z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))
    mu = torch.bmm(input_fea, z_)
    mu = _l2norm(mu, dim=1)
    sims = torch.bmm(mu.permute(0, 2, 1), mu)
    sims = sims.numpy()
    print('Stage {}/{}'.format(i, 3))
    # print("Current Mu: ", mu)



print('done')


