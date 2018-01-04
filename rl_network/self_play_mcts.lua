--
-- Created by HgS_1217_
-- Date: 2017/11/27
--

local utils = require 'utils.utils'
local common = require 'common.common'
local RLPlayer = require 'rl_network.player'
local dcnn_utils = require 'board.dcnn_utils'
local rl_utils = require 'rl_network.utils'
local resnet_rl = require 'resnet.rl_train'
local self_play = require 'rl_network.self_play'


local self_play_mcts = {}
local rl_player

-- Write the SGF file
local function save_sgf_file(res, pb, pw, is_save)
    local footprint = string.format("%s-%s-%s-%s__%d", utils.get_signature(), pb, pw, utils.get_randString(6), rl_player.b._ply)
    local srcSGF = string.format("%s.sgf", footprint)
    local re

    if res.resign_side == common.white then
        re = "B+Resign"
    elseif res.resign_side == common.black then
        re = "W+Resign"
    else
        re = res.score > 0 and string.format("B+%.1f", res.score) or string.format("W+%.1f", -res.score)
    end

    return rl_player:save_sgf(srcSGF, re, pb, pw, is_save)
end

function self_play_mcts.train_per_epoch(opt, epoch)
    local sgf_dataset = {}

    for batch = 1, opt.num_games_per_epoch do
        print(string.format("Train game: %d - %d/%d", epoch, batch, opt.num_games_per_epoch))
        rl_player:clear_board()
        while true do
            local s, str, res = rl_player:g()
            if not s then
                break
            elseif str == "resign" then
                local sgf = save_sgf_file(res, "resnet", "resnet", opt.sgf_save)
                table.insert(sgf_dataset, sgf)
                break
            end
        end
    end

    resnet_rl.train_on_the_fly(opt.model, sgf_dataset, string.format("rl%04d", epoch))
end

function self_play_mcts.train(callbacks, opt)
    rl_player = RLPlayer(callbacks, opt)

    for epoch = 1, opt.num_epoches do
        local old_model = opt.model:clone()
        self_play_mcts.train_per_epoch(opt, epoch)

        local play_opt, opt1, opt2 = rl_utils.train_play_init(old_model, opt.model,
            string.format("resnet_rl%04d", epoch - 1), string.format("resnet_rl%04d", epoch))

        local old_win, new_win, differential = self_play.play(opt1, opt2, play_opt)

        print(string.format('old_win = %d, new_win = %d, differential = %d', old_win, new_win, differential))

        if differential > 0 then
            opt.model = old_model  -- TODO: Save or give up the new model
        end
    end

    rl_player:quit()
end

function self_play_mcts.play_one_step(opt, b)
    local x, y
    if string.sub(opt.codename, 1, 6) == "resnet" then
        x, y = rl_utils.play_with_cnn(b, b._next_player, opt.model)
    else
        x, y = dcnn_utils.sample(opt, b, b._next_player)
    end

    return rl_player:play(x, y, b._next_player) -- this will not work
end

function self_play_mcts.play_mcts(callbacks, opt)
    rl_player = RLPlayer(callbacks, opt)
    for i=1, opt.num_games do
        print(string.format("Playing: %d/%d", i, opt.num_games))
        rl_player:clear_board()

        local mcts_side, pb, pw
        if i % 2 == 1 then
            mcts_side = common.black
            pb, pw = opt.codename_mcts, opt.codename
        else
            mcts_side = common.white
            pw, pb = opt.codename_mcts, opt.codename
        end

        while true do
            local s, str, res
            if rl_player.b._next_player == mcts_side then
                s, str, res = rl_player:g()
            else
                s, str, res = self_play_mcts.play_one_step(opt, rl_player.b)
            end

            if not s then
                print("\n" .. str .. "\n")
                break
            elseif str == "resign" then
                local _, score = rl_player:final_score()
                print("\nFinal Score: " .. score .. "\n")
                save_sgf_file(res, pb, pw, true)
                break
            end
        end
    end
    rl_player:quit()
end

return self_play_mcts