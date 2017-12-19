--
-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant 
-- of patent rights can be found in the PATENTS file in the same directory.
--

local ffi = require 'ffi'
local utils = require('utils.utils')
local goutils = require 'utils.goutils'
local board = require 'board.board'
local common = require("common.common")
local dp = require('board.default_policy')

local symbols, s = utils.ffi_include(paths.concat(common.script_path(), "../common/package.h"))

local num_first_move = tonumber(symbols.NUM_FIRST_MOVES)
local move_normal = tonumber(symbols.MOVE_NORMAL)
local move_simple_ko = tonumber(symbols.MOVE_SIMPLE_KO)
local move_tactical = tonumber(symbols.MOVE_TACTICAL)

local hostname = utils.get_hostname()

local util_pkg = { }

function util_pkg.init(max_batch)
    util_pkg.def_policy = dp.new()

    local params = dp.new_params(true)
    dp.set_params(util_pkg.def_policy, params)

    util_pkg.t_received = { }

    local boards = ffi.new("MBoard*[?]", max_batch)
    local moves = ffi.new("MMove*[?]", max_batch)
    local anchor = {}  -- prevent gc
    for i = 1, max_batch do
        local b = ffi.new("MBoard")
        table.insert(anchor, b)
        boards[i-1] = b

        local m = ffi.new("MMove")
        table.insert(anchor, m)
        moves[i-1] = m
    end
    util_pkg.anchor = anchor

    -- These two are open interface so that other could use. 
    util_pkg.boards = boards
    util_pkg.moves = moves
end

function util_pkg.prepare_move(k, sort_prob, sort_index)
    utils.dprint("Start sending move")

    local mmove = util_pkg.moves[k - 1]
    -- Note that unlike in receive_move, k is no longer batch_idx, since we skip a batch slot if it does not see a board situation.
    local mboard = util_pkg.boards[k - 1]
    local player = mboard.board._next_player

    mmove.t_sent = mboard.t_sent
    mmove.t_received = util_pkg.t_received[k]
    mmove.t_replied = common.wallclock()
    mmove.hostname = hostname
    mmove.player = player
    mmove.b = mboard.b
    mmove.seq = mboard.seq
    utils.dprint("Send b = %x, seq = %d, k = %d", tonumber(mmove.b), tonumber(mmove.seq), k)
    mmove.has_score = common.FALSE

    --
    -- Deal with tactical moves if there are any.
    -- Index i keeps the location where the next move is added to the list.
    local i = 0
    utils.dprint("Add tactical moves")
    local tactical_moves = dp.get_candidate_moves(util_pkg.def_policy, mboard.board)

    for l = 1, #tactical_moves do
        local x = tactical_moves[l][1]
        local y = tactical_moves[l][2]
        mmove.xs[i] = x
        mmove.ys[i] = y
        -- Make the confidence small but not zero.
        mmove.probs[i] = 0.01
        mmove.types[i] = move_tactical
        utils.dprint("   Move (%d, %d), move = %s, type = Tactical move", x, y, goutils.compose_move_gtp(x, y, tonumber(mmove.player)))
        i = i + 1
    end

    -- Check if each move is valid or not, if not, go to the next move.
    -- Index j is the next move to be read from CNN.
    utils.dprint("Add CNN moves")
    local j = 1
    while i < num_first_move and j <= common.board_size * common.board_size do
        local x, y = goutils.moveIdx2xy(sort_index[j])
        local check_res, comments = goutils.check_move(mboard.board, x, y, player)
        if check_res then
            -- The move is all right.
            mmove.xs[i] = x
            mmove.ys[i] = y
            mmove.probs[i] = sort_prob[j]
            if board.is_move_giving_simple_ko(mboard.board, x, y, player) then
               mmove.types[i] = move_simple_ko
            else
               mmove.types[i] = move_normal
            end
            utils.dprint("   Move (%d, %d), ind = %d, move = %s, conf = (%f), type = %s",
                x, y, sort_index[j], goutils.compose_move_gtp(x, y, tonumber(mmove.player)), sort_prob[j], mmove.types[i] == move_simple_ko and "Simple KO move" or "Normal Move")

            i = i + 1
        else
            utils.dprint("   Skipped Move (%d, %d), ind = %d, move = %s, conf = (%f), Reason = %s",
                x, y, sort_index[j], goutils.compose_move_gtp(x, y, tonumber(player)), sort_prob[j], comments)
        end
        j = j + 1
    end
    -- Put zeros if there is not enough move.
    utils.dprint("Zero padding moves")
    while i < num_first_move do
        mmove.xs[i] = 0
        mmove.ys[i] = 0
        mmove.probs[i] = 0
        i = i + 1
    end
    return mmove
end

local gc_count = 0
local gc_interval = 50
function util_pkg.sparse_gc()
    gc_count = gc_count + 1
    if gc_count == gc_interval then
        collectgarbage()
        gc_count = 0
    end
end

function util_pkg.free()
    dp.free(util_pkg.def_policy)
end

return util_pkg