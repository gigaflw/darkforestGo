--
-- Created by HgS_1217_
-- Date: 2017/11/27
--

local utils = require("utils.utils")
local common = require("common.common")
local RLPlayer = require("rl_network.rl_player_mcts")
local dcnn_utils = require("board.dcnn_utils")
local rl_utils = require("rl_network.rl_utils")

local self_play_mcts = {}
local rl_player

-- Write the SGF file
local function save_sgf_file(res, pb, pw)
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

    rl_player:save_sgf(srcSGF, re, pb, pw)
end

function self_play_mcts.train(callbacks, opt)
    rl_player = RLPlayer(callbacks, opt)
    for i=1, opt.num_games do
        io.stderr:write("\nCurrent training: " .. i .. "/" .. opt.num_games .. "\n")
        rl_player:clear_board()
        while true do
            local s, str, res = rl_player:g()
            if not s then
                io.stderr:write("\n" .. str .. "\n")
                break
            else
                if str == "resign" then
                    local _, score = rl_player:final_score()
                    io.stderr:write("\nFinal Score: " .. score .. "\n")
                    save_sgf_file(res, opt_internal.codename, opt_internal.codename)
                    break
                end
            end
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

    return rl_player:play(x, y, b._next_player)
end

function self_play_mcts.play_mcts(callbacks, opt)
    rl_player = RLPlayer(callbacks, opt)
    for i=1, opt.num_games do
        io.stderr:write("\nPlaying: " .. i .. "/" .. opt.num_games .. "\n")
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
                io.stderr:write("\n" .. str .. "\n")
                break
            else
                if str == "resign" then
                    local _, score = rl_player:final_score()
                    io.stderr:write("\nFinal Score: " .. score .. "\n")
                    save_sgf_file(res, pb, pw)
                    break
                end
            end
        end
    end
    rl_player:quit()
end

return self_play_mcts