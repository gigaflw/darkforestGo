--
-- Created by HgS_1217_
-- Date: 2017/11/11
--

package.path = package.path .. ';../?.lua'

local self_play = require("rl_network.self_play")
local rl_utils = require("rl_network.rl_utils")
local board = require("board.board")
local pl = require 'pl.import_into'()

local opt = pl.lapp[[
    --codename1            (default "resnet_15")    Code name AI1 for models. If this is not empty then --input will be omitted.
    --codename2            (default "resnet_14")    Code name AI2 for models.
    --sample_step          (default 1)              Sample step, if the step of a game is smaller, it is a bad sample.
    --resign                                        Whether support resign in rl_training.
    --num_games            (default 1)              The number of games to be played.
    --pipe_path            (default "../../dflog")  Pipe path
]]

local dcnn_opt1, dcnn_opt2 = rl_utils.rl_init(opt)

local b = board.new()

self_play.train(b, dcnn_opt1, dcnn_opt2, opt)
