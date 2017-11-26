-- @Author: gigaflw
-- @Date:   2017-11-23 14:25:44
-- @Last Modified by:   gigaflw
-- @Last Modified time: 2017-11-26 21:58:58

doc = [[
    The following script should always be the entrance of the training procedure
]]

local pl = require 'pl.import_into'()
local opt = pl.lapp[[
    ** Training Options  **
    --batch_size            (default 24)         The number of positions in each batch
    --max_batches           (default 2000)         The number of batches in each epoch
    --epoches               (default 10)         The number of epoches
    --epoch_per_display     (default 1)          The number of epoches per displaying result
    --epoch_per_ckpt        (default 10)          The number of epoches per saving checkpoints
    --ckpt_dir              (default './resnet.ckpt') Where to store the checkpoints

    ** Network Options  **
    --n_residual_blocks     (default 2)          The number of residual blocks in the resnet, 19 or 39 according to the thesis
    
    ** Optimizer Options  **
    ** optimizer opt depends on the type of the optimizer, change them in the hard-coded <optim>-opt **
]]

local sgd_opt = {
  learningRate = 0.1,
  learningRateDecay = 1e-4,
  weightDecay = 1e-4,  -- weight decay is exactly L2 regularization
  momentum = 0.9
} -- these key names are in camel-case because torch.optim library require them to be so

local resnet = require 'resnet.resnet'
local get_dataloader = require 'resnet.dataloader'
local Trainer = require 'resnet.trainer'

local net = resnet.create_model(opt)
local crit = resnet.create_criterion()

local dataloader = get_dataloader('train', opt.batch_size)

require 'optim'
local trainer = Trainer(net, crit, optim.sgd, sgd_opt, opt)
-- trainer:load()
trainer:train(dataloader)
