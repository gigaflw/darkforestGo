--
-- Created by HgS_1217_
-- Date: 2017/11/28
--

local goutils = require 'utils.goutils'
local utils = require('utils.utils')
local common = require("common.common")
local sgfloader = require 'utils.sgf'
local board = require 'board.board'
local om = require 'board.ownermap'
local dp_simple = require('board.default_policy')
local dp_pachi = require('pachi_tactics.moggy')
local dp_v2 = require('board.pattern_v2')
local pl = require 'pl.import_into'()
local class = require "class"
local rl_player = class("RLPlayerMCTS")

local def_policy = dp_pachi.new()
local ownermap = om.new()
local moves = {}

local function verify_player(b, player)
    if player ~= b._next_player then
        local supposed_player = (b._next_player == common.white and 'W' or 'B')
        local curr_player = (player == common.white and 'W' or 'B')
        print(string.format("Wrong player! The player is supposed to be %s but actually is %s...", supposed_player, curr_player))
        return false
    else
        return true
    end
end

-- Init the RLPlayer
function rl_player:__init(callbacks, opt)
    self.b, self.board_initialized = board.new(), false
    self.cbs = callbacks
    self.name = "rl_player_MCTS"
    self.version = "1.0"
    self.ownermap = om.new()
    -- Opt
    local default_opt = {
        win_rate_thres = 0.0,
        default_policy = 'v2',
        default_policy_pattern_file = '../models/playout-model.bin',
        default_policy_temperature = 0.125,
        default_policy_sample_topn = -1,
        save_sgf_per_move = false,
    }

    if opt then
        self.opt = utils.add_if_nonexist(pl.tablex.deepcopy(opt), default_opt)
    else
        self.opt = default_opt
    end

    local rule = (opt and opt.rule == "jp") and board.japanese_rule or board.chinese_rule
    self.rule = opt.rule

    if self.opt.default_policy == 'v2' then
        self.dp = dp_v2
        self.def_policy = self.dp.init(self.opt.default_policy_pattern_file, rule)
        self.dp.set_sample_params(self.def_policy, self.opt.default_policy_sample_topn, self.opt.default_policy_temperature)
    elseif self.opt.default_policy == 'pachi' then
        self.dp = dp_pachi
        self.def_policy = self.dp.new(rule)
    elseif self.opt.default_policy == 'simple' then
        self.dp = dp_simple
        self.def_policy = self.dp.new(rule)
    end

    io.stderr:write("RLPlayerMCTS")
end

function rl_player:check_resign()
    -- If the score is beyond the threshold, then one side will resign.
    local thres = 10

    -- Fast complete the game using pachi tactics, then compute the final score
    local score, _, _, scores = om.util_compute_final_score(
        self.ownermap, self.b, self.val_komi + self.val_handi, nil,
        function (b, max_depth) return self.dp.run(self.def_policy, b, max_depth, false) end
    )
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

-- Write sgf
function rl_player:add_to_sgf_history(x, y, player)
    table.insert(self.sgf_history, { x, y, player })
    if self.opt.save_sgf_per_move then
        self:save_sgf(string.format("game-%d.sgf", self.b._ply - 1))
    end
end

-- Save the current history to sgf.
function rl_player:save_sgf(filename, opt, re)
    local f = io.open(filename, "w")
    if not f then
        return false, "file " .. filename .. " cannot be opened"
    end

    local date = utils.get_current_date()
    local header = {
        komi = self.val_komi,
        handi = self.val_handi,
        rule = self.rule,
        player_b = opt_evaluator.codename,
        player_w = opt_evaluator.codename,
        date = date,
        result = re
    }
    f:write(sgfloader.sgf_string(header, self.sgf_history))
    f:close()
    io.stderr:write("Sgf " .. filename .. " saved.\n")
    return true
end

function rl_player:clear_board()
    board.clear(self.b)
    self.board_initialized = true
    self.board_history = { }
    self.sgf_history = { }

    self.val_komi = 6.5
    self.val_handi = 0
    self.cbs.new_game()

    return true
end

function rl_player:score(show_more)
    if self.val_komi == nil or self.val_handi == nil then
        return false, "komi or handi is not set!"
    end
    -- Computing final score could be cpu hungry, so we need to stop computation if possible
    if self.cbs.thread_switch then
        self.cbs.thread_switch("off")
    end

    local score, livedead, territory, scores = om.util_compute_final_score(
        self.ownermap, self.b, self.val_komi + self.val_handi, nil,
        function (b, max_depth) return self.dp.run(self.def_policy, b, max_depth, false) end
    )

    local min_score = scores:min()
    local max_score = scores:max()
    local stones = om.get_territorylist(territory)

    io.stderr:write(string.format("Score (%s): %f, Playout min: %f, Playout max: %f, #dame: %d", self.opt.default_policy, score, min_score, max_score, #stones.dames));
    if show_more then
        -- Show the deadstone.
        local dead_stones = om.get_deadlist(livedead)
        local dead_stones_info = table.concat(dead_stones.b_str, " ") .. " " .. table.concat(dead_stones.w_str, " ")
        io.stderr:write("Deadstones info:")
        io.stderr:write(dead_stones_info)
        om.show_deadstones(self.b, livedead)

        io.stderr:write("Black prob:")
        om.show_stones_prob(self.ownermap, common.black)
        io.stderr:write("White prob:")
        om.show_stones_prob(self.ownermap, common.white)
    end

    if self.cbs.thread_switch then
        self.cbs.thread_switch("on")
    end
    return true, tostring(score), false, { score = score, min_score = min_score, max_score = max_score, num_dame = #stones.dames, livedead = livedead }
end

function rl_player:g()
    return self:genmove(self.b._next_player)
end

function rl_player:genmove(player)

    if not self.board_initialized then
        return false, "Board should be initialized!!"
    end
    if player == nil then
        return false, "Player should not be null"
    end
    if not verify_player(self.b, player) then
        return false, "Invalid move!"
    end

    -- Save the history.
    table.insert(self.board_history, board.copyfrom(self.b))

    -- Do not pass until after 140 ply.
    -- After that, if enemy pass then we pass.
    if self.b._ply >= 140 and goutils.coord_is_pass(self.b._last_move) then
        -- If the situation has too many dames, we don't pass.
        local _, _, _, stats = self:score()
        if stats.num_dame < 5 then
            -- Play pass here.
            board.play(self.b, 0, 0, player)
            if self.cbs.move_receiver then
                self.cbs.move_receiver(0, 0, player)
            end
            return true, "resign", {
                resign_side = 0,
                score = stats.score,
                min_score = stats.min_score,
                max_score = stats.max_score
            }
        end
    end

    if self.opt.resign and self.b._ply >= 140 and self.b._ply % 20 == 1 then
        io.stderr:write("Check whether we have screwed up...")
        local resign_thres = 10
        local _, _, _, scores = self:score()
        if (player == common.white and scores.min_score > resign_thres) or (player == common.black and scores.max_score < -resign_thres) then
            return true, "resign", {
                resign_side = player,
                score = scores.score,
                min_score = scores.min_score,
                max_score = scores.max_score
            }
        end
        if scores.min_score == scores.max_score and scores.max_score == scores.score then
            -- The estimation is believed to be absolutely correct.
            if (player == common.white and scores.score > 0.5) or (player == common.black and scores.score < -0.5) then
                return true, "resign", {
                    resign_side = player,
                    score = scores.score,
                    min_score = scores.min_score,
                    max_score = scores.max_score
                }
            end
        end
    end

    -- Call move predictor to get the move.
    local xf, yf, win_rate = self.cbs.move_predictor(self.b, player)

    if xf == nil then
        io.stderr:write("Warning! No move is valid!")
        -- Play pass here.
        xf, yf = 0, 0
    end

    local move = goutils.compose_move_gtp(xf, yf)

    -- Actual play this move
    if not board.play(self.b, xf, yf, player) then
        error("Illegal move from move_predictor! move: " .. move)
    end

    self.cbs.adjust_params_in_game(self.b)

    self:add_to_sgf_history(xf, yf, player)

    self.win_rate = win_rate

    return true, move, win_rate
end

function rl_player:final_score()
    local res, _, _, stats = self:score()

    if not res then
        return false, "error in computing score"
    end
    local score = stats.score

    local s =  score > 0 and string.format("B+%.1f", score) or string.format("W+%.1f", -score)
    return true, s
end

function rl_player:quit()
    if self.cbs.quit_func then
        self.cbs.quit_func()
    end
    return true, "Byebye!", true
end

return rl_player