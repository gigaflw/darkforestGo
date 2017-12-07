-- @Author: gigaflw
-- @Date:   2017-12-03 16:01:28
-- @Last Modified by:   gigaflw
-- @Last Modified time: 2017-12-03 18:47:36

local CNNPlayerV2 = require 'cnnPlayerV2.cnnPlayerV2Framework'
local playoutv2 = require('mctsv2.playout_multithread')
local common = require("common.common")
local board = require 'board.board'
local utils = require('utils.utils')

local function set_playout_params_from_opt()
    local p = playoutv2.params
    local tp = playoutv2.tree_params

    p.print_search_tree = common.TRUE
    p.pipe_path = 'mctspipe'
    p.tier_name = 'testing'
    p.server_type = playoutv2.server_local
    p.verbose = 1 -- 1 | 2 | 3
    p.num_gpu = 1
    p.dynkomi_factor = 0.0
    p.cpu_only = common.FALSE
    p.rule = board.chinese_rule

    tp.max_depth_default_policy = 10000
    tp.max_send_attempts = 3
    tp.verbose = 1
    tp.time_limit = 10
    tp.num_receiver = 1
    tp.sigma = 0.05
    tp.use_pondering = false
    tp.use_cnn_final_score = common.FALSE
    tp.final_mixture_ratio = 0.5
    tp.min_ply_to_use_cnn_final_score = 100

    tp.num_tree_thread = 1
    tp.rcv_acc_percent_thres = 80
    tp.rcv_max_num_move = 20
    tp.rcv_min_num_move = 1
    tp.decision_mixture_ratio = 0.5
    tp.single_move_return = common.FALSE

    tp.default_policy_choice = playoutv2.dp_table['v2']
    tp.pattern_filename = "models/playout-model.bin"

    tp.use_online_model = common.FALSE
    tp.online_model_alpha = 0
    tp.online_prior_mixture_ratio = 0
    tp.use_rave = common.FALSE
    tp.use_async = common.FALSE
    tp.expand_n_thres = 0
    tp.num_virtual_games = 0
    tp.percent_playout_in_expansion = 0

    tp.default_policy_sample_topn = -1
    tp.default_policy_temperature = 0.125
    tp.use_old_uct = common.FALSE
    tp.use_sigma_over_n = common.FALSE
    tp.num_playout_per_rollout = 1
end

local tr
local callbacks = {
    new_game = function()
        set_playout_params_from_opt()
        if tr then
            playoutv2.restart(tr)
        else
            tr = playoutv2.new({
                rollout = 1000,
                dcnn_rollout_per_move = 1000,
                rollout_per_move = 1000,
            })
        end
    end,
    move_predictor = function (b, player)
        local m = playoutv2.play_rollout(tr, nil, b)
        return m.x + 1, m.y + 1, m.win_rate
    end,
    move_receiver = function(x, y, player)
        playoutv2.prune_xy(tr, x, y, player, nil)
    end
}

local opt2 = {
    rule = 'cn',
    win_rate_thres = 0.0,   -- resign threshold
    exec = '',              -- initial cmd
    setup_board = '',
    default_policy = 'v2',
    default_policy_pattern_file =  "models/playout-model.bin",
    default_policy_temperature = 0.125,
    default_policy_sample_topn = -1,
    save_sgf_per_move = false
}

-- local cnnplayer = CNNPlayerV2("CNNPlayerV2MCTS", "go_player_v2_mcts", "1.0", callbacks, opt2)
b = board.new()

-- cnnplayer:clear_board()
board.clear(b)
callbacks.new_game()

-- cnnplayer:play('b', 'd4', false) -- call move_receiver to tell search thread to prune
board.play(b, 4, 4, common.black)
playoutv2.prune_xy(tr, 4, 4, common.black, nil)

-- cnnplayer:genmove('w')
m = playoutv2.play_rollout(tr, nil, b) -- the Move class in playout_common.h
print(m.x, m.y, m.player)

board.show(b, 'last_move')
if tr then playoutv2.free(tr) end
