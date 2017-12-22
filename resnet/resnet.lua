-- @Author: gigaflower
-- @Date:   2017-11-21 07:34:01
-- @Last Modified by:   gigaflw
-- @Last Modified time: 2017-12-22 10:33:44

local nn = require 'nn'
local nninit = require 'nninit'

torch.setdefaulttensortype('torch.FloatTensor')

local Conv = nn.SpatialConvolution
local Acti -- to de decided by opt
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
    for _, name in pairs{'n_res', 'n_channel', 'n_feature', 'activation'} do
        assert(opt[name], "ResNet: No opt."..name.." assigned")
    end
    assert(nn[opt.activation], 'unknown activation '..opt.activation)
    local n_residual_blocks = opt.n_res -- 19 or 39 according to the thesis
    local n_conv_channel = opt.n_channel -- 256 according to the thesis
    local n_feature = opt.n_feature
    if opt.acti_param then
        Acti = function () return nn[opt.activation](opt.acti_param, true) end
        -- true means values will be calculated in place
    else
        Acti = function () return nn[opt.activation](true) end
    end

    ---------------------------
    -- Residual Block & Tower
    ---------------------------

    -- The basic residual layer block for 18 and 34 layer network, and the CIFAR networks
    local function residualBlock()
        local s = nn.Sequential()
            :add(Conv(n_conv_channel, n_conv_channel, 3, 3, 1, 1, 1, 1))
            :add(BatchNorm(n_conv_channel))
            :add(Acti())
            :add(Conv(n_conv_channel, n_conv_channel, 3, 3, 1, 1, 1, 1))
            :add(BatchNorm(n_conv_channel))

        return nn.Sequential()
        :add(nn.ConcatTable()
            :add(s)
            :add(nn.Identity()))
        :add(nn.CAddTable(true))
        :add(Acti())
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
            :add(Acti())
            :add(nn.View(19*19*2))
            :add(Linear(19*19*2, 19*19+1))
    end

    local function valueHead()
        return nn.Sequential()
            :add(Conv(n_conv_channel, 1, 1, 1, 1, 1)) -- batch_size x 1 x 19 x 19
            :add(BatchNorm(1))
            :add(Acti())
            :add(nn.View(19*19))
            :add(Linear(19*19, 256))
            :add(Acti())
            :add(Linear(256, 1))
            :add(nn.Sigmoid())
    end

    ---------------------------
    --  The ResNet model
    ---------------------------
    local model = nn.Sequential()
        :add(Conv(n_feature, n_conv_channel, 3, 3, 1, 1, 1, 1)) -- batch_size x n_conv_channel x 19 x 19
        :add(BatchNorm(n_conv_channel))
        :add(Acti())
        :add(residualTower(n_residual_blocks))
        :add(nn.ConcatTable()
            :add(policyHead())
            :add(valueHead()))

    ---------------------------
    --  Init parameters
    -- according to fb.resnet.torch
    ---------------------------
    for _, m in pairs(model:findModules('nn.SpatialConvolution')) do
        m:init('weight', nninit.xavier):init('bias', nninit.constant, 0)
    end
    for _, m in pairs(model:findModules('nn.SpatialBatchNormalization')) do
        m:init('weight', nninit.constant, 1):init('bias', nninit.constant, 0)
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
        :add(nn.BCECriterion(), opt.value_weight)

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
