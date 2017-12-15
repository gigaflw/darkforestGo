--
-- Created by HgS_1217_
-- Date: 2017/11/11
--

local utils = require("utils.utils")

utils.require_torch()
utils.require_cutorch()

cutorch.setDevice(2)

local board = require("board.board")
local om = require("board.ownermap")
local dp = require("pachi_tactics.moggy")
local dcnn_utils = require("board.dcnn_utils")
local sgf = require("utils.sgf")
local common = require("common.common")
local rl_utils = require("rl_network.rl_utils")
local pl = require 'pl.import_into'()

local def_policy = dp.new()
local ownermap = om.new()

local self_play = {}

function self_play.dp_run(b, max_depth)
    return dp.run(def_policy, b, max_depth, false)
end

function self_play.check_resign(b, opt)
    -- If the score is beyond the threshold, then one side will resign.
    local thres = 10

    -- Fast complete the game using pachi tactics, then compute the final score
    local score, _, _, scores = om.util_compute_final_score(ownermap, b, opt.komi + opt.handi, nil, self_play.dp_run)
    local min_score, max_score = scores:min(), scores:max()
    local resign_side

    if min_score > thres then
        resign_side = common.white
    elseif max_score < -thres then
        resign_side = common.black
    elseif min_score == max_score and max_score == score then
        if score > 0.5 then
            resign_side = common.white
        elseif score < -0.5 then
            resign_side = common.black
        end
    end

    return resign_side, score, min_score, max_score
end

function self_play.play_one_game(b, dcnn_opt1, dcnn_opt2, opt)
    -- One game of self play.
    local moves = {}

    while true do

        local m = {}
        -- Resign if one side loses too much.
        if opt.resign and b._ply >= 140 and b._ply % 20 == 1 then
            local resign_side, score, min_score, max_score = self_play.check_resign(b, opt)
            if opt.debug then
                print(string.format('score = %.1f, min_score = %.1f, max_score = %.1f', score, min_score, max_score))
            end
            if resign_side then
                return {
                    moves = moves,
                    resign_side = resign_side,
                    score = score,
                    min_score = min_score,
                    max_score = max_score
                }
            else
                m['C'] = string.format('score = %.1f, min_score = %.1f, max_score = %.1f', score, min_score, max_score)
            end
        end

        -- Generate move
        local dcnn_opt = b._next_player == common.black and dcnn_opt1 or dcnn_opt2
        local x, y
        if string.sub(dcnn_opt.codename, 1, 6) == "resnet" then
            x, y = rl_utils.play_with_cnn(b, b._next_player, dcnn_opt.model.net)
        else
            x, y = dcnn_utils.sample(dcnn_opt, b, b._next_player)
        end

--        print("!".."\t"..tostring(x).."\t"..tostring(y).. "\n")

        if x == nil then
            local player_str = b._next_player == common.white and 'W' or 'B'
            m[player_str] = ''
            x, y = 0, 0
        else
            local player_str, coord_str = sgf.compose_move(x, y, b._next_player)
            m[player_str] = coord_str
        end

        table.insert(moves, m)

        board.play(b, x, y, b._next_player)

--        board.show(b, 'last_move')

        if board.is_game_end(b) then
            break
        end

--        if b._ply > 10 then break end
    end

    local _, score, min_score, max_score = self_play.check_resign(b, opt)
    return {
        moves = moves,
        resign_side = 0,
        score = score,
        min_score = min_score,
        max_score = max_score
    }
end

function self_play.train(b, dcnn_opt1, dcnn_opt2, opt)
    for i = 1, opt.num_games do
        print(string.format("Play game: %d/%d", i, opt.num_games))
        board.clear(b)
        local res = self_play.play_one_game(b, dcnn_opt1, dcnn_opt2, opt)
        if #res.moves <= opt.sample_step then
            print(string.format("Bad sample --- moves: %d, sample_step: %d", #res.moves, opt.sample_step))
        else
            local footprint = string.format("%s-%s__%d", utils.get_signature(), utils.get_randString(6), b._ply)
            local srcSGF = string.format("%s.sgf", footprint)
            local f = assert(io.open(srcSGF, 'w'))
            local re

            if res.resign_side == common.white then
                re = "B+Resign"
            elseif res.resign_side == common.black then
                re = "W+Resign"
            else
                re = res.score > 0 and string.format("B+%.1f", res.score) or string.format("W+%.1f", -res.score)
            end

            local date = utils.get_current_date()
            local header = {
                result = re,
                player_b = opt.codename1,
                player_w = opt.codename2,
                date = date,
                komi = opt.komi
            }

            f:write(sgf.sgf_string(header, res.moves))
            f:close()
        end
        collectgarbage()
        collectgarbage()
    end

    dp.free(def_policy)
    om.free(ownermap)
end

return self_play
