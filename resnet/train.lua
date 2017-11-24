-- @Author: gigaflw
-- @Date:   2017-11-23 14:25:44
-- @Last Modified by:   gigaflw
-- @Last Modified time: 2017-11-24 16:23:36

local resnet = require 'resnet.resnet'
local get_dataloader = require 'resnet.dataloader'
local Trainer = require 'resnet.trainer'

local opt = {
    data_augmentation = false,
    batch_size = 24,
    max_batches = 20
}

local sgd_config = {
  learningRate = 1,  
  learningRateDecay = 1e-4,
  weightDecay = 1e-3,
  momentum = 1e-4
} -- these key names should not be changed


net = resnet.create_model()
crit = resnet.create_criterion()

dataloader = get_dataloader('test', opt.max_batches)
dataloader.load_game(1)

require 'optim'
trainer = Trainer(net, crit, optim.sgd, sgd_config)
trainer:train(dataloader, opt.max_batches)
