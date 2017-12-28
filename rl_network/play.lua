--
-- Created by HgS_1217_
-- Date: 2017/12/16
--

package.path = package.path .. ';../?.lua'

local self_play = require 'rl_network.self_play'
local rl_utils = require 'rl_network.utils'
local pl = require 'pl.import_into'()
local utils = require 'utils.utils'

local opt = pl.lapp[[
    --model1               (default "darkfores2")   Path name to AI2's model file. If this is not empty then --input will be omitted.
    --model2               (default "")             Path name to AI2's model file.
    --max_ply              (default 700)            End game in advance
    --at_random                                     Select moves according to probability, in stead of choosing the move with highest prob
    --sample_step          (default -1)             If the step of a game is less than the threshold, it is a bad sample.
    --resign                                        Whether support resign in rl_training.
    --num_games            (default 2)              The number of games to be playe.
    --pipe_path            (default "../../dflog")  Pipe path
    --device               (default 3)
    --sgf_dir              (default "../dataset/sgf") Where to save sgf
]]

if opt.model1:match("darkfores.+") ~= nil or opt.model2:match("darkfores.+") ~= nil then
    require 'cudnn'
end

if pl.path.exists("/dev/nvidiactl") then
    require 'cunn'
    require 'cutorch'
    cutorch.setDevice(opt.device)
else
    require 'nn'
end

local dcnn_opt1, dcnn_opt2 = rl_utils.play_init(opt)

local win1, win2, score = self_play.play(dcnn_opt1, dcnn_opt2, opt)
print(string.format("model1 wins %.2f%%, %.2f on average", win1/opt.num_games * 100, score / opt.num_games))
self_play.free()
