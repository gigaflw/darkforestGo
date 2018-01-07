-- @Author: gigaflw
-- @Date:   2018-01-04 19:40:49
-- @Last Modified by:   gigaflw
-- @Last Modified time: 2018-01-05 23:17:44

local common = require 'common.common'
local board = require 'board.board'
local utils = require 'utils.utils'

local mcts = require 'mctsv2.playout_multithread'

local TREE -- will be something returned by `mcts.new()`
local OPT = {}

local function init(mcts_opt)
    for k, v in pairs(mcts_opt) do
        OPT[k] = v
    end
end

local tree_json_signature
local tree_json_cnt = 0
local function prepare_prefix()
    if OPT.tree_to_json then
        local prefix = paths.concat(OPT.pipe_path, tree_json_signature, string.format("mcts_%04d", tree_json_cnt))
        tree_json_cnt = tree_json_cnt + 1
        return prefix
    end
end

local function set_mcts_opt()
    mcts.params.print_search_tree = OPT.print_tree and common.TRUE or common.FALSE
    mcts.params.pipe_path = OPT.pipe_path
    mcts.params.tier_name = OPT.tier_name
    mcts.params.server_type = OPT.server_type == "local" and mcts.server_local or mcts.server_cluster
    mcts.params.verbose = OPT.verbose
    mcts.params.num_gpu = OPT.num_gpu
    mcts.params.dynkomi_factor = OPT.dynkomi_factor
    mcts.params.cpu_only = OPT.cpu_only and common.TRUE or common.FALSE
    mcts.params.rule = OPT.rule == "jp" and board.japanese_rule or board.chinese_rule

    -- Whether to use heuristic time manager. If so, then (total time is info->common_params->heuristic_tm_total_time)
    if OPT.heuristic_tm_total_time > 0 then
        local alpha = 0.75;
        mcts.params.heuristic_tm_total_time = OPT.heuristic_tm_total_time
        mcts.params.max_time_spent = 4 * alpha * OPT.heuristic_tm_total_time / (mcts.thres_ply2 + mcts.thres_ply3 - mcts.thres_ply1);
        mcts.params.min_time_spent = mcts.min_time_spent;
    end

    mcts.tree_params.max_depth_default_policy = OPT.dp_max_depth
    mcts.tree_params.max_send_attempts = OPT.max_send_attempts
    mcts.tree_params.verbose = OPT.verbose
    mcts.tree_params.time_limit = OPT.time_limit
    mcts.tree_params.num_receiver = OPT.num_gpu
    mcts.tree_params.sigma = OPT.sigma
    mcts.tree_params.use_pondering = OPT.use_pondering
    mcts.tree_params.use_cnn_final_score = OPT.use_cnn_final_score and common.TRUE or common.FALSE
    mcts.tree_params.final_mixture_ratio = OPT.final_mixture_ratio
    mcts.tree_params.min_ply_to_use_cnn_final_score = OPT.min_ply_to_use_cnn_final_score

    mcts.tree_params.num_tree_thread = OPT.num_tree_thread
    mcts.tree_params.rcv_acc_percent_thres = OPT.acc_prob_thres * 100.0
    mcts.tree_params.rcv_max_num_move = OPT.max_num_move
    mcts.tree_params.rcv_min_num_move = OPT.min_num_move
    mcts.tree_params.decision_mixture_ratio = OPT.decision_mixture_ratio
    mcts.tree_params.single_move_return = OPT.single_move_return and common.TRUE or common.FALSE

    mcts.tree_params.default_policy_choice = mcts.dp_table[OPT.default_policy]
    mcts.tree_params.pattern_filename = OPT.default_policy_pattern_file

    mcts.tree_params.use_online_model = math.abs(OPT.online_model_alpha) < 1e-6 and common.FALSE or common.TRUE
    mcts.tree_params.online_model_alpha = OPT.online_model_alpha
    mcts.tree_params.online_prior_mixture_ratio = OPT.online_prior_mixture_ratio
    mcts.tree_params.use_rave = OPT.use_rave and common.TRUE or common.FALSE
    mcts.tree_params.use_async = OPT.use_async and common.TRUE or common.FALSE
    mcts.tree_params.expand_n_thres = OPT.expand_n_thres
    mcts.tree_params.num_virtual_games = OPT.num_virtual_games
    mcts.tree_params.percent_playout_in_expansion = OPT.percent_playout_in_expansion

    mcts.tree_params.default_policy_sample_topn = OPT.default_policy_sample_topn
    mcts.tree_params.default_policy_temperature = OPT.default_policy_temperature
    mcts.tree_params.use_old_uct = OPT.use_old_uct and common.TRUE or common.FALSE
    mcts.tree_params.use_sigma_over_n = OPT.use_sigma_over_n and common.TRUE or common.FALSE
    mcts.tree_params.num_playout_per_rollout = OPT.num_playout_per_rollout
end

-----------------
--  Callbacks  --
-----------------
local callbacks = {}
function callbacks.set_komi(komi, handi)
    if komi ~= nil then
        mcts.set_params(TREE, { komi = komi })
    end
    -- for large handi game, cnn need to give more candidate, so mcts could possibly be more aggressive
    local changed_params
    if handi and handi <= -5 then
        changed_params = { dynkomi_factor=1.0 }
    end
    if changed_params then
        if mcts.set_params(TREE, changed_params) then
            mcts.print_params(TREE)
        end
    end
end

function callbacks.adjust_params_in_game(b, isCanada)
    -- When we are at the end of game, pay attention to local tactics.
    if not OPT.expand_search_endgame then return end
    local changed_params
    if isCanada then -- enter canada time setting, so lower the rollout number
        local min_rollout = 7500
        changed_params = {num_rollout = min_rollout, num_rollout_per_move = min_rollout}
    else
        if b._ply >= 230 then
            -- Try avoid blunder if there is any.
            -- changed_params = { rcv_max_num_move = 7, rcv_min_num_move = 3 }
        elseif b._ply >= 150 then
            changed_params = { rcv_max_num_move = 5 }
        end
    end
    if changed_params then
        if mcts.set_params(TREE, changed_params) then
            mcts.print_params(TREE)
        end
    end
end

function callbacks.new_game()
    set_mcts_opt(OPT)

    if TREE then
        mcts.restart(TREE)
    else
        local rs = {
            rollout = OPT.rollout,
            dcnn_rollout_per_move = (OPT.dcnn_rollout == -1 and OPT.rollout or OPT.dcnn_rollout),
            rollout_per_move = OPT.rollout
        }

        TREE = mcts.new(rs)
    end
    tree_json_cnt = 0
    tree_json_signature = utils.get_signature()
    io.stderr:write("New MCTS game, tree_json_signature: " .. tree_json_signature)
    -- os.execute("mkdir -p " .. paths.concat(OPT.pipe_path, tree_json_signature))
    mcts.print_params(TREE)
end

function callbacks.quit_func()
    if TREE then
        mcts.free(TREE)
    end
end

function callbacks.move_predictor(board)
    local prefix = prepare_prefix(OPT)
    local m = mcts.play_rollout(TREE, prefix, board)
    if prefix then io.stderr:write("Save tree to " .. prefix) end
    return m.x + 1, m.y + 1, m.win_rate
end

function callbacks.move_receiver(x, y, player)
    local prefix = prepare_prefix(OPT)
    mcts.prune_xy(TREE, x, y, player, prefix)
end

function callbacks.thread_switch(arg)
    if arg == "on" then
        mcts.thread_on(TREE)
    elseif arg == 'off' then
        mcts.thread_off(TREE)
    else
        io.stderr:write("Command " .. arg .. " is not recognized!")
    end
end

return {
    opt = OPT,
    init = init,
    callbacks = callbacks
}