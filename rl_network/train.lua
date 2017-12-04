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
    --codename1            (default "resnet")       Code name AI1 for models. If this is not empty then --input will be omitted.
    --codename2            (default "resnet")       Code name AI2 for models.
    -f,--feature_type      (default "old")          By default we only rl_network old features. If codename is specified, this is omitted.
    -r,--rank              (default "9d")           We play in the level of rank.
    --sample_step          (default 1)              Sample step, if the step of a game is smaller, it is a bad sample.
    --resign                                        Whether support resign in rl_training.
    --use_local_model                               Whether we just load local model from the current path
    --komi                 (default 7.5)            The komi we used
    --handi                (default 0)              The handicap stones we placed.
    -c,--usecpu                                     Whether we use cpu to run the program.
    --shuffle_top_n        (default 300)            We random choose one of the first n move and play it.
    --debug                                         Wehther we use debug mode
    --num_games            (default 1)             The number of games to be played.
    --sample_step          (default -1)             Sample at a particular step.
    --presample_codename1  (default "resnet")
    --presample_codename2  (default "resnet")
    --presample_ft         (default "old")
    --copy_path            (default "")
    --pipe_path            (default "../../dflog")  Pipe path
]]

-- opt.feature_type and opt.rank are necessary for the game to be played.
local dcnn_opt1, dcnn_opt2 = rl_utils.rl_init(opt)

local b = board.new()

self_play.train(b, dcnn_opt1, dcnn_opt2, opt)
