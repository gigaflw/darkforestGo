-- @Author: gigaflw
-- @Date:   2017-12-12 11:00:34
-- @Last Modified by:   gigaflw
-- @Last Modified time: 2017-12-16 10:13:35

local doc = [[
    API for reinforcement learning version of the training of the resnet.
]]

local pl = require 'pl.import_into'()
local tnt = require 'torchnet'

local Trainer = require 'resnet.trainer'
local resnet = require 'resnet.resnet'
local get_dataloader = require 'resnet.dataloader'

-- this pl.lapp doesn't mean you can call it from command line
-- only to keep in conformity with resnet.train.lua
local default_opt = pl.lapp[[
    --log_file           (default '')       If given, log will be saved

    ** Dataset Options  **
    --dataset_dir        (default './dataset')
    --style              (default 'traverse')   'sample': select samples at random; 'traverse': select data in order
    --batch_size         (default 24)       The number of positions in each batch, 2048 in AlphaGo Zero thesis
    --data_augment                          use rotation/reflection to augment dataset
    --data_pool_size     (default -1)       Use a pool to buffer and shuffle the inputs better
    --verbose                               Whether print data loading detailsv
    --debug                                 If given, no shuffling or augmentation will be performed

    ** Training Options  **
    --max_batches        (default 20)       -1 means each epoch will go through all data
    --epochs             (default 100)      The number of epochs, where all data will trained once
    --epoch_per_ckpt     (default 10)       The number of epochs per saving checkpoints
    --ckpt_dir           (default './resnet.ckpt')    Where to store the checkpoints
    --ckpt_prefix        (default '')       Extra info to be prepended to checkpoint files
    --resume_ckpt        (default '')       Whether resume some checkpoints before training
    --continue                              Whether resume epochs (otherwise epoch will begin with 1)

    *-- for rl training, there is no test. Set the option to prevent undefined behavior --*
    --test_batches       (default 1)        The number of batches when testing
    --epoch_per_test     (default 1)        The number of epochs per testing

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


local function _save_sgf_to_dataset(dataset, name)
    local doc = [[
        Save an array of sgf strings into dataset file in the format of torchnet.IndexedDataset.
    ]]
    local writer = tnt.IndexedDatasetWriter(name..'.idx', name..'.bin', 'table')
    for _, d in pairs(dataset) do writer:add({sgf = d}) end
    writer:close()
end

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

function export.train_on_the_fly(model, dataset, name, opt)
    local doc = [[
        This function is to be called by reinforcement learning code,
        which can train the model with the given dataset on the fly.

        This function is blocking, which means it will return after the training ends (after possibly a long time).

        @param: model:
            a network given by `resnet.resnet.create_model' or `torch.load(<ckpt>).net`
        @param: dataset: 
            an array of sgf strings, given by `sgf.sgf_string`
        @param: name:
            The name for checkpoints and generated dataset, no extension name needed.
            Use epoch or version name.
        @param: opt:
            If not given, the default rl opt will be used.
            If you want to modify the default opt, pass in the returned value of `get_opt`

        demo:
            local resnet_rl = require 'resnet.rl_train'
            local train = resnet_rl.train_on_the_fly

            for e = 1, epochs do
                dataset = {}
                for g = 1, games_per_epoch do
                    < self-play for one game >
                    < insert sgf string into dataset >
                end

                train(model, dataset, string.format("rl%04d", e))
                .. -- will return after training ends; arg `model` will have been updated then
                < select current best model >
            end
    ]]

    if opt == nil then opt = default_opt end
    if opt.ckpt_prefix == '' then opt.ckpt_prefix = name end

    local crit = resnet.create_criterion(opt)

    local dataset_path = paths.concat(opt.dataset_dir, name)
    _save_sgf_to_dataset(dataset, dataset_path)  -- generate .bin & .idx file

    local dataloader = get_dataloader(dataset_path, opt)
    local trainer = Trainer(model, crit, opt, dataloader)

    trainer:train()
end

return export