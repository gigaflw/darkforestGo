--
-- Created by HgS_1217_
-- Date: 2017/11/27
--

local playoutv2 = require('mctsv2.playout_multithread')
local common = require("common.common")
local goutils = require 'utils.goutils'
local board = require 'board.board'
local utils = require 'utils.utils'

utils.require_torch()
utils.require_cutorch()


local callbacks = {}

local tr
local count = 0
local signature

function callbacks:__init(opt, a)
    self.opt = opt
end

local function prepare_prefix(opt)
    if opt.tree_to_json then
        local prefix = paths.concat(opt.pipe_path, signature, string.format("mcts_%04d", count))
        count = count + 1
        return prefix
    end
end

local function set_playout_params_from_opt(opt)
    playoutv2.params.print_search_tree = opt.print_tree and common.TRUE or common.FALSE
    playoutv2.params.pipe_path = opt.pipe_path
    playoutv2.params.tier_name = opt.tier_name
    playoutv2.params.server_type = opt.server_type == "local" and playoutv2.server_local or playoutv2.server_cluster
    playoutv2.params.verbose = opt.verbose
    playoutv2.params.num_gpu = opt.num_gpu
    playoutv2.params.dynkomi_factor = opt.dynkomi_factor
    playoutv2.params.cpu_only = opt.cpu_only and common.TRUE or common.FALSE
    playoutv2.params.rule = opt.rule == "jp" and board.japanese_rule or board.chinese_rule

    -- Whether to use heuristic time manager. If so, then (total time is info->common_params->heuristic_tm_total_time)
    if opt.heuristic_tm_total_time > 0 then
        local alpha = 0.75;
        playoutv2.params.heuristic_tm_total_time = opt.heuristic_tm_total_time
        playoutv2.params.max_time_spent = 4 * alpha * opt.heuristic_tm_total_time / (playoutv2.thres_ply2 + playoutv2.thres_ply3 - playoutv2.thres_ply1);
        playoutv2.params.min_time_spent = playoutv2.min_time_spent;
    end

    playoutv2.tree_params.max_depth_default_policy = opt.dp_max_depth
    playoutv2.tree_params.max_send_attempts = opt.max_send_attempts
    playoutv2.tree_params.verbose = opt.verbose
    playoutv2.tree_params.time_limit = opt.time_limit
    playoutv2.tree_params.num_receiver = opt.num_gpu
    playoutv2.tree_params.sigma = opt.sigma
    playoutv2.tree_params.use_pondering = opt.use_pondering
    playoutv2.tree_params.use_cnn_final_score = opt.use_cnn_final_score and common.TRUE or common.FALSE
    playoutv2.tree_params.final_mixture_ratio = opt.final_mixture_ratio
    playoutv2.tree_params.min_ply_to_use_cnn_final_score = opt.min_ply_to_use_cnn_final_score

    playoutv2.tree_params.num_tree_thread = opt.num_tree_thread
    playoutv2.tree_params.rcv_acc_percent_thres = opt.acc_prob_thres * 100.0
    playoutv2.tree_params.rcv_max_num_move = opt.max_num_move
    playoutv2.tree_params.rcv_min_num_move = opt.min_num_move
    playoutv2.tree_params.decision_mixture_ratio = opt.decision_mixture_ratio
    playoutv2.tree_params.single_move_return = opt.single_move_return and common.TRUE or common.FALSE

    playoutv2.tree_params.default_policy_choice = playoutv2.dp_table[opt.default_policy]
    playoutv2.tree_params.pattern_filename = opt.default_policy_pattern_file

    playoutv2.tree_params.use_online_model = math.abs(opt.online_model_alpha) < 1e-6 and common.FALSE or common.TRUE
    playoutv2.tree_params.online_model_alpha = opt.online_model_alpha
    playoutv2.tree_params.online_prior_mixture_ratio = opt.online_prior_mixture_ratio
    playoutv2.tree_params.use_rave = opt.use_rave and common.TRUE or common.FALSE
    playoutv2.tree_params.use_async = opt.use_async and common.TRUE or common.FALSE
    playoutv2.tree_params.expand_n_thres = opt.expand_n_thres
    playoutv2.tree_params.num_virtual_games = opt.num_virtual_games
    playoutv2.tree_params.percent_playout_in_expansion = opt.percent_playout_in_expansion

    playoutv2.tree_params.default_policy_sample_topn = opt.sample_topn
    playoutv2.tree_params.default_policy_temperature = opt.default_policy_temperature
    playoutv2.tree_params.use_old_uct = opt.use_old_uct and common.TRUE or common.FALSE
    playoutv2.tree_params.use_sigma_over_n = opt.use_sigma_over_n and common.TRUE or common.FALSE
    playoutv2.tree_params.num_playout_per_rollout = opt.num_playout_per_rollout
end

function callbacks:set_komi(komi, handi)
    if komi ~= nil then
        playoutv2.set_params(tr, { komi = komi })
    end
    -- for large handi game, cnn need to give more candidate, so mcts could possibly be more aggressive
    local changed_params
    if handi and handi <= -5 then
        changed_params = { dynkomi_factor=1.0 }
    end
    if changed_params then
        if playoutv2.set_params(tr, changed_params) then
            playoutv2.print_params(tr)
        end
    end
end

function callbacks:adjust_params_in_game(b, isCanada)
    -- When we are at the end of game, pay attention to local tactics.
    if not self.opt.expand_search_endgame then return end
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
        if playoutv2.set_params(tr, changed_params) then
            playoutv2.print_params(tr)
        end
    end
end

function callbacks:on_time_left(sec_left, num_moves)
    playoutv2.set_time_left(tr, sec_left, num_moves)
end

function callbacks:new_game()
    set_playout_params_from_opt(self.opt)

    if tr then
        playoutv2.restart(tr)
    else
        local rs = {
            rollout = self.opt.rollout,
            dcnn_rollout_per_move = (self.opt.dcnn_rollout == -1 and self.opt.rollout or self.opt.dcnn_rollout),
            rollout_per_move = self.opt.rollout
        }

        tr = playoutv2.new(rs)
    end
    count = 0
    signature = utils.get_signature()
    io.stderr:write("New MCTS game, signature: " .. signature)
    os.execute("mkdir -p " .. paths.concat(self.opt.pipe_path, signature))
    playoutv2.print_params(tr)
end

function callbacks:quit_func()
    if tr then
        playoutv2.free(tr)
    end
end

function callbacks:move_predictor(b, player)
    local prefix = prepare_prefix(self.opt)
    local m = playoutv2.play_rollout(tr, prefix, b)
    if prefix then io.stderr:write("Save tree to " .. prefix) end
    return m.x + 1, m.y + 1, m.win_rate
end

function callbacks:move_receiver(x, y, player)
    local prefix = prepare_prefix(self.opt)
    playoutv2.prune_xy(tr, x, y, player, prefix)
end

function callbacks:peek_simulation(num_simulation)
    return playoutv2.set_params(tr, { min_rollout_peekable = num_simulation })
end

function callbacks:move_peeker(b, player, topk)
    return playoutv2.peek_rollout(tr, topk, b)
end

function callbacks:undo_func(b, undone_move)
    if goutils.coord_is_pass(undone_move) then
        playoutv2.undo_pass(tr, b)
    else
        playoutv2.set_board(tr, b)
    end
end

function callbacks:set_board(b)
    playoutv2.set_board(tr, b)
end

function callbacks:thread_switch(arg)
    if arg == "on" then
        playoutv2.thread_on(tr)
    elseif arg == 'off' then
        playoutv2.thread_off(tr)
    else
        io.stderr:write("Command " .. arg .. " is not recognized!")
    end
end

function callbacks:set_move_history(history)
    for _, h in pairs(history) do
        playoutv2.add_move_history(tr, unpack(h))
    end
end

function callbacks:set_verbose_level(verbose_level)
    if playoutv2.set_params(tr, { verbose = verbose_level }) then
        playoutv2.print_params(tr)
    end
end

return callbacks