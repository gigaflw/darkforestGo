--
-- Created by HgS_1217_
-- Date: 2017/11/27
--

local utils = require("utils.utils")
local board = require("board.board")
local om = require("board.ownermap")
local dcnn_utils = require("board.dcnn_utils")
local sgf = require("utils.sgf")
local common = require("common.common")
local dp_simple = require('board.default_policy')
local dp_pachi = require('pachi_tactics.moggy')
local dp_v2 = require('board.pattern_v2')
local sgfloader = require("utils.sgf")
local pl = require 'pl.import_into'()

local RLPlayer = require("rl_network.rl_player_mcts")

local self_play_mcts = {}
local rl_player

-- Write the SGF file
local function save_sgf_file(res, opt)
    local footprint = string.format("%s-%s__%d", utils.get_signature(), utils.get_randString(6), rl_player.b._ply)
    local srcSGF = string.format("%s.sgf", footprint)
    local re

    if res.resign_side == common.white then
        re = "B+Resign"
    elseif res.resign_side == common.black then
        re = "W+Resign"
    else
        re = res.score > 0 and string.format("B+%.1f", res.score) or string.format("W+%.1f", -res.score)
    end

    rl_player:save_sgf(srcSGF, opt, re)
end

function self_play_mcts.train(callbacks, opt)
    rl_player = RLPlayer(callbacks, opt)
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
                save_sgf_file(res, opt)
                break
            end
        end
    end
    rl_player:quit()
end

return self_play_mcts