-- @Author: gigaflower
-- @Date:   2017-11-21 07:34:01
-- @Last Modified by:   gigaflw
-- @Last Modified time: 2017-11-24 16:22:19

local nn = require 'nn'

local Conv = nn.SpatialConvolution
local ReLU = nn.ReLU
local Linear = nn.Linear
local BatchNorm = nn.SpatialBatchNormalization

local function create_model()
    local nResiudalBlocks = 3 -- 19 or 39 according to the thesis

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
    local function residualTower(nResiudalBlocks)
        local s = nn.Sequential()
        for i= 1, nResiudalBlocks do
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
    --  The ResNet ImageNet model
    ---------------------------
    local model = nn.Sequential()
        :add(Conv(17, 256, 3, 3, 1, 1, 1, 1)) -- batch_size x 256 x 19 x 19
        :add(BatchNorm(256))
        :add(ReLU(true))
        :add(residualTower(nResiudalBlocks))
        :add(nn.ConcatTable()
            :add(policyHead())
            :add(valueHead()))

    return model
end

local function create_criterion()
    -- Make this a function should custom options be necessary
    -- usage:
    --  c = create_criterion()
    --  pred = { < a 362-d vecto >, < a scalar > }
    --  label = { < a integer between [1, 362] >, < -1 or 1 > }
    --  loss = c:forward(pred, label)
    return nn.ParallelCriterion()
        :add(nn.CrossEntropyCriterion())
        :add(nn.MSECriterion())  
end

local function _test()
    crit = create_criterion()

    fake = torch.rand(2, 17, 19, 19)
    p, v = table.unpack(model:forward(fake))
    print(#p)
end

return {
    create_model = create_model,
    create_criterion = create_criterion
}
