--
-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant 
-- of patent rights can be found in the PATENTS file in the same directory.
--

local ffi = require 'ffi'
local utils = require 'utils.utils'
local goutils = require 'utils.goutils'
local board = require 'board.board'
local common = require 'common.common'
local dp = require 'board.default_policy'

local symbols, s = utils.ffi_include(paths.concat(common.script_path(), "../common/package.h"))

local NUM_FIRST_MOVES = tonumber(symbols.NUM_FIRST_MOVES)
local MOVE_NORMAL = tonumber(symbols.MOVE_NORMAL)
local MOVE_SIMPLE_KO = tonumber(symbols.MOVE_SIMPLE_KO)
local MOVE_TACTICAL = tonumber(symbols.MOVE_TACTICAL)

local hostname = utils.get_hostname()

local util = { }

function util.init(max_batch)
    util.def_policy = dp.new()

    local params = dp.new_params(true)
    dp.set_params(util.def_policy, params)

    util.t_received = { }

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
    util.anchor = anchor

    -- These two are open interface so that other could use. 
    util.boards = boards
    util.moves = moves
end

function util.prepare_move(ind, sort_prob, sort_index, use_dp)
    local doc = [[
        set moves[ind] so that it is prepared to be sent back
        @params: sort_prob:
            Tensor, sorted probability given by neuron network
        @params: sort_index:
            Tensor, corresponding move index
        @params: use_dp:
            Whether considere the tactical moves computed by default policy.

        e.g.
            sort_prob = [0.8, 0.1, ...., 0]
            sort_index = [4, 1, ...., 0]
            this means the neuron network predict the position `4` to be the best move with probability 0.8

        Following moves will be considered:
            * All moves computed from default policy, with prob 0.01, if use_dp = true
            * The top k moves selected given by neuron network
            the number of moves is bounded by NUM_FIRST_MOVES
    ]]

    utils.dprint("Start sending move")

    local mmove = util.moves[ind - 1]
    -- Note that unlike in receive_move, k is no longer batch_idx, since we skip a batch slot if it does not see a board situation.
    local mboard = util.boards[ind - 1]
    local player = mboard.board._next_player

    mmove.t_sent = mboard.t_sent
    mmove.t_received = util.t_received[ind]
    mmove.t_replied = common.wallclock()
    mmove.hostname = hostname
    mmove.player = player
    mmove.b = mboard.b
    mmove.seq = mboard.seq
    utils.dprint("Send b = %x, seq = %d, ind = %d", tonumber(mmove.b), tonumber(mmove.seq), ind)
    mmove.has_score = common.FALSE

    local good_move_cnt = 0

    -- Deal with tactical moves if there are any.
    -- Index i keeps the location where the next move is added to the list.
    if use_dp then
        utils.dprint("Add tactical moves")
        local tactical_moves = dp.get_candidate_moves(util.def_policy, mboard.board)

        for _, move in pairs(tactical_moves) do
            local x, y = move[1], move[2]
            mmove.xs[good_move_cnt], mmove.ys[good_move_cnt] = x, y
            mmove.probs[good_move_cnt] = 0.01 -- Make the confidence small but not zero.
            mmove.types[good_move_cnt] = MOVE_TACTICAL
            utils.dprint("   Move (%d, %d), move = %s, type = Tactical move",
                x, y, goutils.compose_move_gtp(x, y, tonumber(mmove.player)))
            good_move_cnt = good_move_cnt + 1
        end
    end

    -- Check if each move is valid or not, if not, go to the next move.
    -- Index i is the next move to be read from CNN.
    utils.dprint("Add CNN moves")
    for i = 1, 361 do
        if good_move_cnt >= NUM_FIRST_MOVES then break end

        local move_idx, prob = sort_index[i], sort_prob[i]
        local x, y = goutils.moveIdx2xy(move_idx)
        local check_res, comments = goutils.check_move(mboard.board, x, y, player)

        if check_res then
            -- The move is all right.
            mmove.xs[good_move_cnt] = x
            mmove.ys[good_move_cnt] = y
            mmove.probs[good_move_cnt] = prob
            mmove.types[good_move_cnt] = board.is_move_giving_simple_ko(mboard.board, x, y, player) and MOVE_SIMPLE_KO or MOVE_NORMAL
            good_move_cnt = good_move_cnt + 1

            utils.dprint("   Move (%d, %d), ind = %d, move = %s, conf = (%f), type = %s",
                x, y, move_idx, goutils.compose_move_gtp(x, y, tonumber(mmove.player)), prob, mmove.types[i] == MOVE_SIMPLE_KO and "Simple KO move" or "Normal Move")
        else
            utils.dprint("   Skipped Move (%d, %d), ind = %d, move = %s, conf = (%f), Reason = %s",
                x, y, move_idx, goutils.compose_move_gtp(x, y, tonumber(player)), prob, comments)
        end
    end

    -- Put zeros if there is not enough move.
    if good_move_cnt < NUM_FIRST_MOVES then utils.dprint("Zero padding moves") end
    for i = good_move_cnt, NUM_FIRST_MOVES - 1 do
        mmove.xs[i] = 0
        mmove.ys[i] = 0
        mmove.probs[i] = 0
    end

    return mmove
end

local gc_count = 0
local gc_interval = 50
function util.sparse_gc()
    gc_count = gc_count + 1
    if gc_count == gc_interval then
        collectgarbage()
        gc_count = 0
    end
end

function util.free()
    dp.free(util.def_policy)
end

return util