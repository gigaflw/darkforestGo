-- @Author: gigaflw
-- @Date:   2017-12-06 17:03:09
-- @Last Modified by:   gigaflw
-- @Last Modified time: 2017-12-06 18:35:07

local pl = require 'pl.import_into'()


if pl.path.exists("/dev/nvidiactl") then
    use_gpu = true
    require 'cunn'
    require 'cudnn'
    cutorch.setDevice(4)
    Tensor = torch.CudaTensor
else
    use_gpu = false
    Tensor = torch.FloatTensor
end

require 'nn'

if arg[1] == '--n_res' then
    local resnet = require 'resnet.resnet'
    net = resnet.create_model({n_res = tonumber(arg[2]), use_gpu=use_gpu})
else
    local ckpt = arg[1] or 'resnet.ckpt/latest.params'
    net = torch.load(ckpt)
    print(net.opt)
    print(net.optim_state)
    net = net.net
end

input = Tensor(1, 17, 19, 19):zero()
input[{1, 17}] = 1 -- black

out = net:forward(input)
prob = nn.SoftMax():forward(out[1]:float())
p, m = (-prob):topk(5)

print('win rate for B: ', out[2][1])
print('max/min/avg/std: ', out[1]:max(), out[1]:min(), out[1]:mean(), out[1]:std())

function idx2xy(idx)
    local x = math.floor((idx - 1) / 19) + 1
    local y = (idx - 1) % 19 + 1
    return x, y
end

for k = 1, 5 do
    print('p x y: ', -p[k], idx2xy(m[k]))
end
