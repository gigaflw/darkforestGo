-- @Author: gigaflower
-- @Date:   2017-11-21 07:34:01
-- @Last Modified by:   gigaflw
-- @Last Modified time: 2017-11-21 20:09:25

local nn = require 'nn'

local Conv = nn.SpatialConvolution
local ReLU = nn.ReLU
local Linear = nn.Linear
local BatchNorm = nn.SpatialBatchNormalization

local function createModel()
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
  -- in: 19 x 19 x 256
  -- out: 19 x 19 x 256, not changed
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
      :add(Conv(256, 2, 1, 1, 1, 1)) -- 2 x 19 x 19
      :add(BatchNorm(2))
      :add(ReLU(true))
      :add(nn.View(-1))
      :add(Linear(19*19*2, 19*19+1))
  end

  local function valueHead()
    return nn.Sequential()
      :add(Conv(256, 1, 1, 1, 1, 1)) -- 1 x 19 x 19
      :add(BatchNorm(1))
      :add(ReLU(true))
      :add(nn.View(-1))
      :add(Linear(19*19, 256))
      :add(ReLU(true))
      :add(Linear(256, 1))
      :add(nn.Tanh())
  end

  ---------------------------
  --  The ResNet ImageNet model
  ---------------------------
  local model = nn.Sequential()
    :add(Conv(17, 256, 3, 3, 1, 1, 1, 1)) -- 256 x 19 x 19
    :add(BatchNorm(256))
    :add(ReLU(true))
    :add(residualTower(nResiudalBlocks))
    :add(nn.ConcatTable()
      :add(policyHead())
      :add(valueHead()))

  return model
end

return createModel
-- model = createModel()
-- fake = torch.rand(1, 17, 19, 19)
-- ret = model:forward(fake)

