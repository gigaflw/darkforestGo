-- @Author: gigaflower
-- @Date:   2017-11-21 07:34:01
-- @Last Modified by:   gigaflw
-- @Last Modified time: 2017-11-27 19:07:26

local nn = require 'nn'
local nninit = require 'nninit'

torch.setdefaulttensortype('torch.FloatTensor')

local Conv = nn.SpatialConvolution
local ReLU = nn.ReLU
local Linear = nn.Linear
local BatchNorm = nn.SpatialBatchNormalization

local function create_model(opt)
    assert(opt.n_res, "ResNet: No opt.n_res assigned")
    local n_residual_blocks = opt.n_res -- 19 or 39 according to the thesis

    ---------------------------
    -- Residual Block & Tower
    ---------------------------

    -- The basic residual layer block for 18 and 34 layer network, and the CIFAR networks
    local function residualBlock()
        local s = nn.Sequential()
            :add(Conv(256, 256, 3, 3, 1, 1, 1, 1))
            :add(BatchNorm(256))
            :add(ReLU(true))
            :add(Conv(256, 256, 3, 3, 1, 1, 1, 1))
            :add(BatchNorm(256))

        return nn.Sequential()
        :add(nn.ConcatTable()
            :add(s)
            :add(nn.Identity()))
        :add(nn.CAddTable(true))
        :add(ReLU(true))
    end

    -- Creates count residual blocks with specified number of features
    -- in: batch_size x 19 x 19 x 256
    -- out: batch_size x 19 x 19 x 256, identical shape
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
            :add(Conv(256, 2, 1, 1, 1, 1)) -- batch_size x 2 x 19 x 19
            :add(BatchNorm(2))
            :add(ReLU(true))
            :add(nn.View(19*19*2))
            :add(Linear(19*19*2, 19*19+1))
    end

    local function valueHead()
        return nn.Sequential()
            :add(Conv(256, 1, 1, 1, 1, 1)) -- batch_size x 1 x 19 x 19
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
        :add(Conv(17, 256, 3, 3, 1, 1, 1, 1)) -- batch_size x 256 x 19 x 19
        :add(BatchNorm(256))
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
