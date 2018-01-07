--
-- Created by HgS_1217_
-- Date: 2017/11/28
--

local goutils = require 'utils.goutils'
local utils = require 'utils.utils'
local common = require 'common.common'
local sgfloader = require 'utils.sgf'
local board = require 'board.board'
local om = require 'board.ownermap'
local pl = require 'pl.import_into'()

local class = require 'class'
local player = class("Player")

function player:__init(callbacks, opt)
    self.opt = pl.tablex.deepcopy(opt)

    self.b, self.board_initialized = board.new(), false
    self.cbs = callbacks
    self.name = "maim"
    self.version = "1.0"
    self.ownermap = om.new()

    local rule = (opt and opt.rule == "jp") and board.japanese_rule or board.chinese_rule
    self.rule = opt.rule

    self:_init_dp()
    self:clear_board()
    print("maim "..self.version.." on")
end

function player:_init_dp()
    -- init default policy
    local dp_simple = require 'board.default_policy'
    local dp_pachi = require 'pachi_tactics.moggy'
    local dp_v2 = require 'board.pattern_v2'

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
end

function player:mainloop()
    while true do
        local line = io.read()
        if line == nil then break end
        local ret, quit = self:parse_command(line)
        print(ret..'\n\n')
        io.flush()
        if quit == true then break end
    end
end

function player:parse_command(line, mode)
    if line == nil then return false end
    local content = pl.utils.split(line)

    if #content == 0 then return false end

    local cmdid = ''
    if string.match(content[1], "%d+") then
        cmdid = table.remove(content, 1)
    end

    local command = table.remove(content, 1)
    local successful, outputstr, quit

    if type(player['cmd_'..command]) ~= 'function' then
        print("Warning: Ignoring unknown command - " .. line)
    else
        successful, outputstr, quit = player['cmd_'..command](self, unpack(content))
    end
    local ret
    if successful then
        if outputstr == nil then outputstr = '' end
        ret = string.format("=%s %s\n", cmdid, outputstr)
    else
        ret = string.format("?%s ??? %s\n", cmdid, outputstr)
    end
    return ret, quit
end

----------------------------------------------------
-- gtp command functions
--- @args: all args are string
--- @return: all gtp functions need to return a three-tuple
---     < succeed, outputstr, quit >
---   where succeed: boolean, whether the cmd is valid
---         outputstr:  string, optional
---         quit: boolean, optional, whether to quit the main loop
----------------------------------------------------
function player:cmd_protocol_version()
    return true, '2'
end

function player:cmd_name()
    return true, self.name
end

function player:cmd_version()
    return true, self.version
end

function player:cmd_known_command(cmd)
    return true, type(player['cmd_'..cmd]) == 'function' and "true" or "false"
end

function player:cmd_list_commands()
    local cmds = {}
    for name, _ in pairs(getmetatable(player)) do
        if name:sub(1, 4) == 'cmd_' then
            table.insert(cmds, name:sub(5))
        end
    end
    return true, table.concat(cmds, '\n')
end

function player:cmd_quit()
    self:quit()
    return true, "Byebye!", true
end

function player:cmd_boardsize(board_size)
    if board_size == nil then return false end
    local s = tonumber(board_size)
    if s ~= board.get_board_size(b) then
        error(string.format("Board size %d is not supported!", s))
    end
    return true
end

function player:cmd_clear_board()
    self:clear_board()
    return true
end

function player:cmd_komi(komi_val)
    self.val_komi = tonumber(komi_val)
    if self.cbs.set_komi then self.cbs.set_komi(komi_val + self.val_handi) end
    return true
end

function player:cmd_play(player, coord)
    local x, y, player = goutils.parse_move_gtp(coord, player)
    if x == nil then x, y = 0, 0 end
    local valid, str = self:play(player, x, y)
    board.show(self.b, 'last_move')
    return valid, str
end

function player:cmd_genmove(player)
    player = (player:lower() == 'w' or player:lower() == 'white') and common.white or common.black
    local valid, str = self:genmove(player)
    board.show(self.b, 'last_move')
    return valid, str
end

function player:cmd_show_board()
    board.show(self.b, "last_move")
    return true
end
----------------------------------------------------
-- gtp command functions end
----------------------------------------------------

--------------------------
-- Util functions
--------------------------
function player:score(show_more)
    local doc = [[
        Compute the score of the current game by play the game to the end with `default policy` several times
        This can be TIME CONSUMING, modify the default policy parameter to control the performance

        @returns: < bool_successful >, < stat >
            where stat is table like: {
                score = < average score of BLACK side >,
                min_score = < min score >,
                max_score = < max score >,
                max_score = < max score >,
                num_dame = < how many dame we have now >,
                livedead = < a 19x19 char tensor denoting the dead/live stones>
            }
    ]]
    if self.val_komi == nil or self.val_handi == nil then
        return false, "komi or handi is not set!"
    end
    -- Computing final score could be cpu hungry, so we need to stop computation if possible
    if self.cbs.thread_switch then self.cbs.thread_switch("off") end

    local score, livedead, territory, scores = om.util_compute_final_score(
        self.ownermap, self.b, self.val_komi + self.val_handi, nil,
        function (b, max_depth) return self.dp.run(self.def_policy, b, max_depth, false) end
    )

    local min_score = scores:min()
    local max_score = scores:max()
    local stones = om.get_territorylist(territory)

    io.stderr:write(string.format("Score (%s): %f, Playout min: %f, Playout max: %f, #dame: %d\n", self.opt.default_policy, score, min_score, max_score, #stones.dames));
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

    if self.cbs.thread_switch then self.cbs.thread_switch("on") end
    return true, { score = score, min_score = min_score, max_score = max_score, num_dame = #stones.dames, livedead = livedead }
end


function player:check_resign(threshold)
    local doc = [[
        Run default policy serveral times to see whether any side is doomed now
        This can be TIME CONSUMING

        When some side will resign:
        a. one side wins/loses at least `threshold` stones
        b. all dp result has same score, which means there can be no change in the score, early exit

        @returns: < bool_do_resign >, < stat >
            where stat is table like: {
                resign_side = [ common.black | common.white ],
                score = < average score of BLACK side >,
                min_score = < min score >,
                max_score = < max score >,
            }
    ]]
    local _, scores = self:score()
    local stat = { score = scores.score, min_score = scores.min_score, max_score = scores.max_score }

    if scores.min_score > resign_thres or scores.max_score < -resign_thres then
        stat.resign_side = scores.min_score > resign_thres and common.white or common.black
        return true, stat
    end
    if scores.min_score == scores.max_score and scores.max_score == scores.score then
        stat.resign_side = scores.score > 0 and common.white or common.black
        return true, stat
    end

    return false
end


function player:play(player, x, y)
    local doc = [[
        @params: player: [ common.black | common.white ]
        @params: x, y: int from 1 to 19, give 0, 0 to denote a pass move
        @return: 
            false, "Invalid move!"  if invalid move
            true, "resign"          if game ends
            true, < move info str>  otherwise
    ]]
    if not self.board_initialized then error("Board should be initialized!!") end
    if not self:verify_player(player) then
        return false, "Invalid move!"
    end

    if not board.play(self.b, x, y, player) then
        io.stderr:write(string.format("Illegal move from the opponent! x = %d, y = %d, player = %d", x, y, player))
        return false, "Invalid move"
    end

    if self.cbs.move_receiver then self.cbs.move_receiver(x, y, player) end
    if self.cbs.adjust_params_in_game then self.cbs.adjust_params_in_game(self.b) end
    self:add_to_sgf_history(x, y, player)

    if board.is_game_end(self.b) then
        return true, "resign"
    end

    return true, string.format("* ply = %d, x = %d, y = %d, player = %d", self.b._ply, x, y, player)
end

function player:g()
    return self:genmove(self.b._next_player)
end

function player:genmove(player)
    local doc == [[
        Generate move according to `callbacks.move_predictor`
        @params: player: [ common.black | common.white ]
        @return:
            false, < error info >   if invalid
            true, "pass"            if we pass
            true, "resign"          if we resign
            true, < move str >      otherwise

    ]]
    if not self.board_initialized then
        return false, "Board should be initialized!!"
    end
    if player == nil then
        return false, "Player should not be null"
    end
    if not self:verify_player(player) then
        return false, "Invalid move!"
    end


    -- Do not pass until after 140 ply.
    -- After that, if enemy pass then we pass.
    if self.b._ply >= 140 and goutils.coord_is_pass(self.b._last_move) then
        -- If the situation has too many dames, we don't pass.
        local _, stats = self:score()
        if stats.num_dame < 5 then
            board.play(self.b, 0, 0, player)
            if self.cbs.move_receiver then self.cbs.move_receiver(0, 0, player) end
            return true, "pass"
        end
    end

    -- Check whether we should resign ...
    if self.opt.resign and self.b._ply >= 140 and self.b._ply % self.opt.resign_step == 1 then
        io.stderr:write("Check whether we have screwed up...")
        local thres = self.opt.resign_thres or 10
        local do_resign, stat_resign = self:check_resign(thres)
        if do_resign then
            return true, "resign", stat_resign
        end
    end

    -- Call move predictor to get the move.
    local t_start = common.wallclock()
    local x, y, win_rate = self.cbs.move_predictor(self.b, player)

    if x == nil then
        io.stderr:write("Warning! No move is valid!")
        x, y = 0, 0 -- Play pass here.
    end

    local move = goutils.compose_move_gtp(x, y)

    -- Actual play this move
    if not board.play(self.b, x, y, player) then
        error("Illegal move from move_predictor! move: " .. move)
    end

    if self.cbs.adjust_params_in_game then self.cbs.adjust_params_in_game(self.b) end  -- FIXME: necessary?
    self:add_to_sgf_history(x, y, player)
    self.win_rate = win_rate

    print(string.format("* Time spent in genmove %d : %.3fs", self.b._ply, common.wallclock() - t_start))

    return true, move, win_rate
end

function player:final_score()
    local res, stats = self:score()

    if not res then
        return false, "error in computing score"
    end
    local score = stats.score

    local s =  score > 0 and string.format("B+%.1f", score) or string.format("W+%.1f", -score)
    return true, s
end

function player:clear_board()
    board.clear(self.b)
    self.board_initialized = true
    self.sgf_history = { }

    self.val_komi = 6.5
    self.val_handi = 0
    if self.cbs.new_game then self.cbs.new_game() end
end

function player:quit()
    if self.cbs.quit_func then self.cbs.quit_func() end
end

function player:verify_player(player)
    if player ~= self.b._next_player then
        local supposed_player = (self.b._next_player == common.white and 'W' or 'B')
        local curr_player = (player == common.white and 'W' or 'B')
        print(string.format("Wrong player! The player is supposed to be %s but actually is %s...", supposed_player, curr_player))
        return false
    else
        return true
    end
end

-- Write sgf
function player:add_to_sgf_history(x, y, player)
    table.insert(self.sgf_history, { x, y, player })
end

-- Save the current history to sgf.
function player:save_sgf(filename, re, pb, pw, is_save)
    local date = utils.get_current_date()
    local header = {
        komi = self.val_komi,
        handi = self.val_handi,
        rule = self.rule,
        player_b = pb,
        player_w = pw,
        date = date,
        result = re
    }

    local res = sgfloader.sgf_string(header, self.sgf_history)

    if is_save then
        local f = io.open(filename, "w")
        if not f then return false, "file " .. filename .. " cannot be opened" end
        f:write(res)
        f:close()
        io.stderr:write("Sgf " .. filename .. " saved.\n")
    end

    return res
end
--------------------------
-- Util function ends
--------------------------

return player