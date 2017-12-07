-- @Author: gigaflower
-- @Date:   2017-11-21 07:34:01
-- @Last Modified by:   gigaflw
-- @Last Modified time: 2017-12-07 22:21:52

local nn = require 'nn'
local nninit = require 'nninit'

torch.setdefaulttensortype('torch.FloatTensor')

local Conv = nn.SpatialConvolution
local ReLU = nn.ReLU
local Linear = nn.Linear
local BatchNorm = nn.SpatialBatchNormalization

local doc = [[
    The resnet model is like this:

             Conv
       Batch normalization
             ReLU
        Residual Tower
         /           \
    Policy Head    Value Head

    where
    Residual Tower: comprise of `opt.n_res` residual blocks
    Policy Head: Conv - BN - ReLU - Linear
    Value  Head: Conv - BN - ReLU - Linear - ReLU - Linear - Tanh

    * input & output:

    input:
        batch_size x feature_plane x 19 x 19
    policy output:
        batch_size x 362
        a probability for positions for next move
        (19*19+1, +1 for `pass` move)
    value output:
        batch_size
        a vector for winning rate in (-1, 1)

    * usage:
    > net = create_model(opt)
    > output = net:forward(input)
    > print(output)
    {
        1: <policy output> 
        2: <Value output>
    }
    > crit = create_criterion(opt)
    > loss = crit:forward(output, label)  -- label should be a table like output
    > print(loss)                         -- loss = policy_loss + value_loss + other_loss
    <a float>
]]

local function create_model(opt)
    assert(opt.n_res, "ResNet: No opt.n_res assigned")
    assert(opt.n_channel, "ResNet: No opt.n_channel assigned")
    local n_residual_blocks = opt.n_res -- 19 or 39 according to the thesis
    local n_conv_channel = opt.n_channel -- 256 according to the thesis

    ---------------------------
    -- Residual Block & Tower
    ---------------------------

    -- The basic residual layer block for 18 and 34 layer network, and the CIFAR networks
    local function residualBlock()
        local s = nn.Sequential()
            :add(Conv(n_conv_channel, n_conv_channel, 3, 3, 1, 1, 1, 1))
            :add(BatchNorm(n_conv_channel))
            :add(ReLU(true))
            :add(Conv(n_conv_channel, n_conv_channel, 3, 3, 1, 1, 1, 1))
            :add(BatchNorm(n_conv_channel))

        return nn.Sequential()
        :add(nn.ConcatTable()
            :add(s)
            :add(nn.Identity()))
        :add(nn.CAddTable(true))
        :add(ReLU(true))
    end

    -- Creates count residual blocks with specified number of features
    -- in: batch_size x n_conv_channel x 19 x 19
    -- out: batch_size x n_conv_channel x 19 x 19, identical shape
    local function residualTower(n_residual_blocks)
        local s = nn.Sequential()
        for i= 1, n_residual_blocks do
            s:add(residualBlock(features, i == 1 and stride or 1))
        end
        return s
    end

    ---------------------------
    --  Policy Head & Value Head
    ---------------------------
    local function policyHead()
        return nn.Sequential()
            :add(Conv(n_conv_channel, 2, 1, 1, 1, 1)) -- batch_size x 2 x 19 x 19
            :add(BatchNorm(2))
            :add(ReLU(true))
            :add(nn.View(19*19*2))
            :add(Linear(19*19*2, 19*19+1))
    end

    local function valueHead()
        return nn.Sequential()
            :add(Conv(n_conv_channel, 1, 1, 1, 1, 1)) -- batch_size x 1 x 19 x 19
            :add(BatchNorm(1))
            :add(ReLU(true))
            :add(nn.View(19*19))
            :add(Linear(19*19, 256))
            :add(ReLU(true))
            :add(Linear(256, 1))
            :add(nn.Tanh())
    end

    ---------------------------
    --  The ResNet model
    ---------------------------
    local model = nn.Sequential()
        :add(Conv(17, n_conv_channel, 3, 3, 1, 1, 1, 1)) -- batch_size x n_conv_channel x 19 x 19
        :add(BatchNorm(n_conv_channel))
        :add(ReLU(true))
        :add(residualTower(n_residual_blocks))
        :add(nn.ConcatTable()
            :add(policyHead())
            :add(valueHead()))

    ---------------------------
    --  Init parameters
    -- according to fb.resnet.torch
    ---------------------------
    for _, pre in pairs{'nn', 'cudnn'} do
        for _, m in pairs(model:findModules(pre..'.SpatialConvolution')) do
            m:init('weight', nninit.xavier):init('bias', nninit.constant, 0)
        end
        for _, m in pairs(model:findModules(pre..'.SpatialBatchNormalization')) do
            m:init('weight', nninit.constant, 1):init('bias', nninit.constant, 0)
        end
    end

    if opt.use_gpu then
        require 'cunn'
        model = model:cuda()
    end

    return model
end

local function create_criterion(opt)
    -- Make this a function should custom options be necessary
    -- usage:
    --  c = create_criterion()
    --  pred = { < a 362-d vecto >, < a scalar > }
    --  label = { < a integer between [1, 362] >, < -1 or 1 > }
    --  loss = c:forward(pred, label)
    crit = nn.ParallelCriterion()
        :add(nn.CrossEntropyCriterion())
        :add(nn.MSECriterion())

    if opt.use_gpu then
        require 'cunn'
        crit = crit:cuda()
    end

    return crit
end

return {
    create_model = create_model,
    create_criterion = create_criterion
}
