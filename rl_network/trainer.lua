-- @Author: gigaflw
-- @Date:   2017-12-19 19:50:51
-- @Last Modified by:   gigaflw
-- @Last Modified time: 2017-12-19 23:24:48

local utils = require 'utils.utils'
local board = require 'board.board'
local om = require 'board.ownermap'
local sgf = require 'utils.sgf'
local common = require 'common.common'
local dp_simple = require 'board.default_policy'
local dp_pachi = require 'pachi_tactics.moggy'
local dp_v2 = require 'board.pattern_v2'
local sgfloader = require 'utils.sgf'
local pl = require 'pl.import_into'()

local RLPlayer = require 'rl_network.player'
local rl_utils = require 'rl_network.utils'
local self_play = require 'rl_network.self_play'

local class = require 'class'
local Trainer = class('rl_network.Trainer')

function Trainer:__init(net, opt, callbacks)
    self.opt = {}
    for k, v in pairs(opt) do self.opt[k] = v end 

    self.net = net
    self.player = RLPlayer(callbacks, opt)
    self.resnet = require 'resnet.rl_train'
end

function Trainer:train()
    opt = self.opt

    self:log('Reinforcement training starts')

    local timer = torch.Timer()

    for e = 1, opt.epoches do

        ----------------------------
        -- generate dataset
        ----------------------------
        local sgf_dataset = {}
        for batch = 1, opt.game_per_epoch do
            self:log(string.format("Train game: %d - %d/%d", e, batch, opt.game_per_epoch))
            local sgf = self:play_one_game()
            sgf_dataset[batch] = sgf
        end

        ----------------------------
        -- supervised training
        ----------------------------
        self.resnet.train_on_the_fly(self.net, sgf_dataset, string.format("rl%04d", epoch))
        self.net = self:get_current_best_model()

        ----------------------------
        -- print result & save ckpt
        ----------------------------
        self:log(string.format("| Epoch %d ends in %.4fs", e, timer:time().real))
        timer:reset()

        self.net:clearState()
        if math.fmod(e, opt.epoch_per_ckpt) == 0 then
            self:save(e, string.format('rl.e%04d.params', e))
        else
            self:save(e, 'rl.latest.params') -- save 'latest.params' every epoch
        end
    end

    self:log("Reinforcement training ends")
end

function Trainer:get_current_best_model()
    -- TODO
    -- decide the stronger model between the old and the new
    -- the stronger will be passed to the next epoch
    -- local play_opt, opt1, opt2 = rl_utils.train_play_init(old_model, opt.model,
    --     string.format("resnet_rl%04d", epoch - 1), string.format("resnet_rl%04d", epoch))

    -- local old_win, new_win, differential = self_play.play(opt1, opt2, play_opt)

    -- print(string.format('old_win = %d, new_win = %d, differential = %d', old_win, new_win, differential))

    -- if differential > 0 then
    --     opt.model = old_model  -- TODO: Save or give up the new model
    -- end
    return self.net
end

function Trainer:play_one_game()
    self.player:clear_board()
    local sgf
    while true do
        local valid, move, res = self.player:g()
        if not valid then
            break
        elseif move == "resign" then
            sgf = self:save_sgf_file(res, "resnet", "resnet", opt.sgf_save)
            break
        end
    end
    return sgf
end

function Trainer:save(epoch, filename)
    local obj = {
        epoch = epoch,
        net = self.net,
        opt = self.opt,
    }
    filename = self.opt.ckpt_prefix..'.'..filename
    torch.save(paths.concat(self.opt.ckpt_dir, filename), obj)
    self:log("checkpoint '"..filename.."' saved")
end

function Trainer:log(message)
    print(message)
    if self.opt.log_file ~= '' then
        local f = io.open(self.opt.log_file, 'a')
        f:write(message..'\n')
        f:close()
    end
end

-- Write the SGF file
function Trainer:save_sgf_file(res, pb, pw, is_save)
    local footprint = string.format("%s-%s-%s-%s__%d", utils.get_signature(), pb, pw, utils.get_randString(6), self.player.b._ply)
    local srcSGF = string.format("%s.sgf", footprint)
    local re

    if res.resign_side == common.white then
        re = "B+Resign"
    elseif res.resign_side == common.black then
        re = "W+Resign"
    else
        re = res.score > 0 and string.format("B+%.1f", res.score) or string.format("W+%.1f", -res.score)
    end

    return self.player:save_sgf(srcSGF, re, pb, pw, is_save)
end

return Trainer
