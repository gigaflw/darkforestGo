
package.path = package.path .. ';../?.lua'

require "torch"

local utils = require 'utils.utils'

utils.require_torch()
utils.require_cutorch()

local model = torch.load("../models/df2.bin")
local board = require 'board.board'
local goutils = require 'utils.goutils'
local common = require("common.common")
local pl = require 'pl.import_into'()
local b = board.new()

local opt = pl.lapp[[
    --codename             (default "darkfores2")   Code name for models. If this is not empty then --input will be omitted.
    -f,--feature_type      (default "old")       By default we only rl_network old features. If codename is specified, this is omitted.
    -r,--rank              (default "9d")        We play in the level of rank.
    --use_local_model                            Whether we just load local model from the current path
    --komi                 (default 7.5)         The komi we used
    --handi                (default 0)           The handicap stones we placed.
    -c,--usecpu            (default 0)           Whether we use cpu to run the program.
    --shuffle_top_n        (default 300)         We random choose one of the first n move and play it.
    --debug                                      Wehther we use debug mode
    --num_games            (default 10)          The number of games to be played.
    --sample_step          (default -1)          Sample at a particular step.
    --presample_codename   (default "darkfores2")
    --presample_ft         (default "old")
    --copy_path            (default "")
]]

local sortProb, sortInd = goutils.play_with_cnn(b, b._next_player, opt, opt.rank, model)

print(sortProb, sortInd)
