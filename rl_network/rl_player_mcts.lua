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
local opt_internal = require("rl_network.rl_cnn_evaluator_opt")
local pl = require 'pl.import_into'()
local class = require "class"

local rl_player = class("RLPlayerMCTS")

local disable_time_left = false
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

    -- set callbacks
    -- possible callback:
    -- 1. move_predictor(board, player)
    --    call move_predictor when the bot is asked to generate a move. This is mandatory.
    -- 2. move_receiver(x, y, player)
    --    call move_receiver when the bot receives opponent moves.
    -- 3. new_game()
    --    When the client receives the comamnd of clear_board
    -- 4. undo_func(prev_board, undone_move)
    --    When the client click undo
    -- 5. set_board(new_board)
    --    Called when setup_board/clear_board is invoked
    -- 6. set_komi(komi)
    --    When komi's set.
    -- 7. quit_func()
    --    When qutting.
    -- 8. thread_switch("on" or "off")
    --    Switch on/off the computation process.
    -- 9. set_move_history(moves) moves = { {x1, y1, player1}, {x2, y2, player2} }..
    --    Set the history of the game. Called when setup_board is invoked.
    --10. set_attention(x_left, y_top, x_right, y_bottom)
    --    Set the attention of the engine (So that the AI will focus on the region more).
    --11. adjust_params_in_game(board_situation)
    --    Depending on the board situation, change the parameters.
    --12. set_verbose_level(level)
    --    Set the verbose level
    --13. on_time_left(sec, num_move)
    --    On time left.

    local valid_callbacks = {
        move_predictor = true,
        move_receiver = true,
        move_peeker = true,
        new_game = true,
        undo_func = true,
        set_board = true,
        set_komi = true,
        quit_func = true,
        thread_switch = true,
        set_move_history = true,
        set_attention = true,
        adjust_params_in_game = true,
        set_verbose_level = true,
        get_value = true,
        on_time_left = true,
        peek_simulation = true
    }

    assert(callbacks)
    assert(callbacks.move_predictor)

    -- Check if there is any misnaming.
    for k, f in pairs(callbacks) do
        if not valid_callbacks[k] then error("The callback function " .. k .. " is not valid") end
        if type(f) ~= 'function' then error("Callback " .. k .. " is not a function!") end
    end

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

    -- default to chinese rule
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

-- Set time left
function rl_player:time_left(color, num_seconds, num_moves)
    local thiscolor = (color:lower() == 'w' or color:lower() == 'white') and common.white or common.black
    if self.mycolor and thiscolor == self.mycolor and num_seconds and num_moves then
        io.stderr:write(string.format("timeleft -- color: %s, num_seconds: %s, num_moves: %s", color, num_seconds, num_moves))
        if self.cbs.on_time_left then
            if not disable_time_left then
                self.cbs.on_time_left(tonumber(num_seconds), tonumber(num_moves))
            else
                print("Time left was disabled")
            end
        end
    else
        io.stderr:write(string.format("enemy timeleft -- color: %s, num_seconds: %s, num_moves: %s", color, num_seconds, num_moves))
    end

    return true
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
        player_b = opt_internal.codename,
        player_w = opt_internal.codename,
        date = date,
        result = re
    }
    f:write(sgfloader.sgf_string(header, self.sgf_history))
    f:close()
    io.stderr:write("Sgf " .. filename .. " saved.\n")
    return true
end

function rl_player:clear_board()
    -- To prevent additional overhead for clear_board twice.
    if not self.board_history or #self.board_history > 0 or self.b._ply > 1 then
        board.clear(self.b)
        self.board_initialized = true
        self.board_history = { }
        self.sgf_history = { }

        -- Default value.
        self.val_komi = 6.5
        self.val_handi = 0
        self.mycolor = nil
        -- Call the new game callback when the board is cleaned.
        if self.cbs.new_game then
            self.cbs.new_game()
        end
    end
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
    local player = self.b._next_player == common.black and 'b' or 'w'
    return self:genmove(player)
end

function rl_player:genmove(player)

    if not self.board_initialized then
        return false, "Board should be initialized!!"
    end
    if player == nil then
        return false, "Player should not be null"
    end
    player = (player:lower() == 'w' or player:lower() == 'white') and common.white or common.black
    if not self.mycolor then
        self.mycolor = player
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
            return true, "pass"
        end
    end

    if self.b._ply >= 140 and self.b._ply % 20 == 1 then
        io.stderr:write("Check whether we have screwed up...")
        local resign_side, score, min_score, max_score = self:check_resign()
        if self.opt.debug then
            print(string.format('score = %.1f, min_score = %.1f, max_score = %.1f', score, min_score, max_score))
        end
        if resign_side then
            return true, "resign", {
                resign_side = resign_side,
                score = score,
                min_score = min_score,
                max_score = max_score
            }
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

--    if self.cbs.move_receiver then
--        self.cbs.move_receiver(xf, yf, player)
--    end

    if board.is_game_end(self.b) then
        local _, score, min_score, max_score = self:check_resign()
        return true, "resign", {
            resign_side = 0,
            score = score,
            min_score = min_score,
            max_score = max_score
        }
    end

    -- Check if we need to adjust parameters in the engine.
    if self.cbs.adjust_params_in_game then
        self.cbs.adjust_params_in_game(self.b)
    end

    self:add_to_sgf_history(xf, yf, player)

    -- Keep this win rate.
    self.win_rate = win_rate

    -- Tell the GTP server we have chosen this move
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