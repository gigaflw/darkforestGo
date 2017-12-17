--
-- Created by HgS_1217_
-- Date: 2017/11/11
--

package.path = package.path .. ';../?.lua'

require("_torch_class_patch")

local self_play = require("rl_network.self_play")
local rl_utils = require("rl_network.rl_utils")
local pl = require 'pl.import_into'()

local opt = pl.lapp[[
    --model_name           (default "resnet_16")    Code name for models. If this is not empty then --input will be omitted.
    --step_threshold       (default 1)              If the step of a game is less than the threshold, it is a bad sample.
    --resign                                        Whether support resign in rl_training.
    --num_epoches          (default 1)              The number of batches in rl_training.
    --num_games_per_epoch  (default 2)              The number of games to be played in an epoch.
    --pipe_path            (default "../../dflog")  Pipe path
    --sgf_save                                      Whether save sgf file per game in rl_training.

    ** GPU Options  **
    --use_gpu            (default true)     No use when there is no gpu devices
    --device             (default 3)        which core to use on a multicore GPU environment
]]

local opt = rl_utils.rl_init(opt)

local util = require 'resnet.util'
opt.use_gpu = opt.use_gpu and util.have_gpu() -- only use gpu when there is one

if opt.use_gpu then
    require 'cutorch'
    cutorch.setDevice(opt.device)
    print('use gpu device '..opt.device)
end

self_play.train(opt)
