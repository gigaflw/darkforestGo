--
-- Created by HgS_1217_
-- Date: 2017/12/16
--

package.path = package.path .. ';../?.lua'

local self_play = require("rl_network.self_play")
local rl_utils = require("rl_network.rl_utils")
local pl = require 'pl.import_into'()

local opt = pl.lapp[[
    --codename1            (default "resnet_17")    Code name AI1 for models. If this is not empty then --input will be omitted.
    --codename2            (default "resnet_18")    Code name AI2 for models.
    --sample_step          (default -1)             If the step of a game is less than the threshold, it is a bad sample.
    --resign                                        Whether support resign in rl_training.
    --num_epoches          (default 1)              The number of batches in rl_training.
    --num_games_per_epoch  (default 1)              The number of games to be played in an epoch.
    --pipe_path            (default "../../dflog")  Pipe path
]]

local dcnn_opt1, dcnn_opt2 = rl_utils.play_init(opt)

self_play.play(dcnn_opt1, dcnn_opt2, opt)
