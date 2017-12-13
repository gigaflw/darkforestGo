-- @Author: gigaflw
-- @Date:   2017-12-12 11:00:34
-- @Last Modified by:   gigaflw
-- @Last Modified time: 2017-12-15 16:44:05

local doc = [[
    API for reinforcement learning version of the training of the resnet.
]]

local pl = require 'pl.import_into'()
local Trainer = require 'resnet.trainer'
local get_dataloader = require 'resnet.dataloader'
-- this doesn't mean you can call it from command line
-- only to keep in conformity with resnet.train.lua
local default_opt = pl.lapp[[
    --test               If true, only run the test epoch
    --log_file           (default '')       If given, log will be saved

    ** Dataset Options  **
    --batch_size         (default 24)       The number of positions in each batch, 2048 in AlphaGo Zero thesis
    --data_augment                          use rotation/reflection to augment dataset
    --verbose                               Whether print data loading detailsv
    --data_pool_size     (default 240)      Use a pool to buffer and shuffle the inputs better
    --debug                                 If given, no shuffling or augmentation will be performed

    ** Training Options  **
    --max_batches        (default 20)       The number of batches in each epoch (commonly 1 epoch means to go through all data, however here it is too large)
    --test_batches       (default 20)       The number of batches when testing
    --epochs             (default 100)      The number of epochs
    --epoch_per_test     (default 1)        The number of epochs per testing
    --epoch_per_ckpt     (default 10)       The number of epochs per saving checkpoints
    --ckpt_dir           (default '')       Where to store the checkpoints
    --resume_ckpt        (default '')       Whether resume some checkpoints before training
    --continue                              Whether resume epochs (otherwise epoch will begin with 1)

    ** GPU Options  **
    --use_gpu
    --device             (default 3)        which core to use on a multicore GPU environment

    ** Network Options  **
    --n_res              (default 2)        The number of residual blocks in the resnet, 19 or 39 according to the thesis
    --n_channel          (default 64)       The number of channels in each residual block, 256 in the thesis
    --n_feature          (default 12)       The number of feature planes, this should be in accordence with feature extraction function in util.lua
    --activation         (default 'ELU')    The type of activation function, 'ReLu' | 'ELU'
    --acti_param         (default 0.1)      Activation parameter, incase activation other than ReLU is used
    --value_weight       (default 1.0)      Loss = Policy Loss + weight * Value Loss

    ** Optimizer Options  **
    --lr                (default 0.1)       learning rate
    --lr_decay          (default 5e-5)      learning rate decay
    --wd                (default 1e-4)      weight decay, exactly L2 regularization
    --momentum          (default 0.9)
]]


local export = {}

function export.get_opt(custom_opt)
    local doc = [[
        Customize default options if you want.
        > default_opt = { a=1, b=2, c=3 }
        > get_opt({ a=10, d=11 })
        { a=10, b=2, c=3, d=11 }
    ]]
    local ret = {}
    for k, v in pairs(default_opt) do ret[k] = v end
    if custom_opt then
        for k, v in pairs(custom_opt) do ret[k] = v end -- custom will override default ones
    end
    return ret
end

function export.train_on_the_fly(model, dataset, opt)
    local doc = [[
        This function is to be called by reinforcement learning code,
        which can train the model with the given dataset on the fly.

        This function is blocking, which means it will return after the training ends (after possibly a long time).

        @param: model:
            a network given by `resnet.resnet.create_model' or `torch.load(<ckpt>).net`
        @param: dataset: 
            the custom dataset, an array in the format of:
            {
                {
                    b = the current board
                    m = moveIdx, an integer in [1, 19*19+1], denoting the right move, the extra '+1' means pass.
                        Should it be a pass, a = 19*19+1
                    p = `common.black` | `common.white`, who is to play
                    w = `common.black` | `common.white`, who wins
                },
                ...
            }
        @param: opt:
            If not given, the default rl opt will be used.
            If you want to modify the default opt, pass in the returned value of `get_opt`

        usage:
            local resnet_rl = require 'resnet.rl_train'
            local train = resnet_rl.train_on_the_fly

            for e = 1, epochs do
                dataset = {}
                for g = 1, games_per_epoch do
                    < self-play for one game >
                    < insert data into dataset >
                end

                train(model, dataset)
                .. -- will return after training ends; arg `model` will have been updated then
                < select current best model >
            end
    ]]

    if opt == nil then opt = default_opt end

    local crit = resnet.create_criterion(opt)
    local dataloader = get_dataloader('custom', opt)
    -- TODO
    local trainer = Trainer(model, crit, opt, dataloader)

    trainer.train()
end

return export