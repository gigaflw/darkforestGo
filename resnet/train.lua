-- @Author: gigaflw
-- @Date:   2017-11-23 14:25:44
-- @Last Modified by:   gigaflw
-- @Last Modified time: 2017-11-27 16:51:39

doc = [[
    The following script should always be the entrance of the training procedure
]]

local pl = require 'pl.import_into'()
local opt = pl.lapp[[
    --test               If true, only run the test epoch

    ** Dataset Options  **
    --batch_size         (default 24)       The number of positions in each batch
    --data_augment       (default true)     use rotation/reflection to augment dataset

    ** Training Options  **
    --max_batches        (default 20)       The number of batches in each epoch (commonly 1 epoch means to go through all data, however here it is too large)
    --test_batches       (default 20)       The number of batches when testing
    --epochs             (default 100)      The number of epochs
    --epoch_per_test     (default 1)        The number of epochs per testing
    --epoch_per_ckpt     (default 10)       The number of epochs per saving checkpoints
    --ckpt_dir           (default './resnet.ckpt')    Where to store the checkpoints
    --resume_ckpt        (default '')       Whether resume some checkpoints before training

    ** GPU Options  **
    --use_gpu

    ** Network Options  **
    --n_residual_blocks  (default 2)        The number of residual blocks in the resnet, 19 or 39 according to the thesis

    ** Optimizer Options  **
    --lr                (default 0.1)       learning rate
    --lr_decay          (default 5e-5)      learning rate decay
    --wd                (default 1e-4)      weight decay, exactly L2 regularization
    --momentum          (default 0.9)
]]

local resnet = require 'resnet.resnet'
local get_dataloader = require 'resnet.dataloader'
local Trainer = require 'resnet.trainer'

local net = resnet.create_model(opt)
local crit = resnet.create_criterion()

local train_dataloader = get_dataloader('train', opt)
local test_dataloader = get_dataloader('test', opt)

local trainer = Trainer(net, crit, opt, train_dataloader, test_dataloader)
if opt.test then
    trainer:test()
else
    trainer:train()
end
