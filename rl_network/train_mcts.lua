--
-- Created by HgS_1217_
-- Date: 2017/11/27
--

require '_torch_class_patch'

local playoutv2 = require 'mctsv2.playout_multithread'
local common = require 'common.common'
local utils = require 'utils.utils'
local board = require 'board.board'
local pl = require 'pl.import_into'()

local opt = pl.lapp[[
    ** Trainer Options **
    --epoches            (default 10)        The number of batches in rl_training.
    --epoch_per_ckpt     (default 1)
    --game_per_epoch     (default 3)         The number of games to be played in an epoch.
    --resign                                 Whether support resign in rl_training.
    --sgf_save                               Whether save sgf file per game in rl_training.
    --log_file           (default 'log.txt')        If given, log will be saved
    --ckpt_dir           (default './rl.ckpt')    Where to store the checkpoints
    --ckpt_prefix        (default '')        Extra info to be prepended to checkpoint files
    --model_filename     (default './resnet.ckpt/latest.cpu.params')        Filename for model

    ** GPU Options **
    --use_gpu            (default true)     No use when there is no gpu devices
    --device             (default 2)        which core to use on a multicore GPU environment

    ** PlayoutV2 Options **
    *** misc ***
    -v,--verbose        (default 1)             The verbose level (1 = critical, 2 = info, 3 = debug)
    --print_tree                                Whether print the search tree.
    --tier_name         (default "ai.go-evaluator")     Tier name
    --tree_to_json                              Whether we save the tree to json file for visualization. Note that pipe_path will be used.

    *** cnn evaluator ***
    --pipe_path         (default ".")           Pipe path
    --server_type       (default "local")       We can choose "local" or "cluster". For open source version, for now "cluster" is not usable.
    --max_send_attempts (default 3)             #attempts to send to the server.
    
    *** tree search ***
    --num_tree_thread   (default 16)            The number of threads used to expand MCTS tree.
    --rollout           (default 2)             How many games are played in one search
    --dcnn_rollout      (default -1)            The number of dcnn rollout we use (If we set to -1, then it is the same as rollout), if cpu_only is set, then dcnn_rollout is not used.
    
    *** default policy **
    --dp_max_depth      (default 10000)         The max_depth of default policy, ignored if patternv2 is used

    --num_gpu           (default 1)             The number of gpus to use for local play.
    --sigma             (default 0.05)          Sigma used to perturb the win rate in MCTS search.
    --use_sigma_over_n                          use sigma / n (or sqrt(nparent/n)). This makes sigma small for nodes with confident win rate estimation.
    --num_virtual_games (default 0)             Number of virtual games we use.
    --acc_prob_thres    (default 0.8)           Accumulated probability threshold. We remove the remove if by the time we see it, the accumulated prob is greater than this thres.
    --max_num_move      (default 20)            Maximum number of moves to consider in each tree node.
    --min_num_move      (default 1)             Minimum number of moves to consider in each tree node.
    --decision_mixture_ratio (default 5.0)      Mixture MCTS count ratio with cnn_confidence.
    --time_limit        (default 1)             Limit time for each move in second. If set to 0, then there is no time limit.
    --win_rate_thres    (default 0.0)           If the win rate is lower than that, resign.
    --use_pondering                             Whether we use pondering
    --exec              (default "")            Whether we run an initial script
    --setup_board       (default "")            Setup board. The argument is "sgfname moveto"
    --dynkomi_factor    (default 0.0)           Use dynkomi_factor
    --num_playout_per_rollout (default 1)       Number of playouts per rollouts.
    --single_move_return                        Use single move return (When we only have one choice, return the move immediately)
    --expand_search_endgame                     Whether we expand the search in end game.
    --default_policy    (default "v2")          The default policy used. Could be "simple", "v2".
    --default_policy_pattern_file (default "models/playout-model.bin") The patter file
    --default_policy_temperature  (default 0.125)   The temperature we use for sampling.
    --online_model_alpha         (default 0.0)      Whether we use online model and its alpha
    --online_prior_mixture_ratio (default 0.0)      Online prior mixture ratio.
    --use_rave                                      Whether we use RAVE.
    --use_cnn_final_score                           Whether we use CNN final score.
    --min_ply_to_use_cnn_final_score (default 100)    When to use cnn final score.
    --final_mixture_ratio            (default 0.5)    The mixture ratio we used.
    --percent_playout_in_expansion   (default 0)      The percent of threads that will run playout when we expand the node. Other threads will block wait.
    --use_old_uct                                     Use old uct
    --use_async                                       Open async model.
    --cpu_only                                        Whether we only use fast rollout.
    --expand_n_thres                 (default 0)      Statistics collected before expand.
    --sample_topn                    (default -1)     If use v2, topn we should sample..
    --rule                           (default jp)     Use JP rule : jp, use CN rule: cn
    --heuristic_tm_total_time        (default 0)      Time for heuristic tm (0 mean you don't use it).
    --min_rollout_peekable           (default 20000)  The command peek will return if the minimal number of rollouts exceed this threshold
    --use_formal_params                               If so, then use formal parameters
    --use_custom_params                               If so, then use custom parameters
]]

local resnet_utils = require 'resnet.utils'
opt.use_gpu = opt.use_gpu and resnet_utils.have_gpu() -- only use gpu when there is one

if opt.use_gpu then
    require 'cutorch'
    cutorch.setDevice(opt.device)
    print('use gpu device '..opt.device)
end

local tr
local count = 0
local signature

local function prepare_prefix()
    if opt.tree_to_json then
        local prefix = paths.concat(opt.pipe_path, signature, string.format("mcts_%04d", count))
        count = count + 1
        return prefix
    end
end

local function set_playout_params_from_opt()
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

-----------------
--  Callbacks  --
-----------------
local callbacks = {}
function callbacks.set_komi(komi, handi)
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

function callbacks.adjust_params_in_game(b, isCanada)
    -- When we are at the end of game, pay attention to local tactics.
    if not opt.expand_search_endgame then return end
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

function callbacks.new_game()
    set_playout_params_from_opt(opt)

    if tr then
        playoutv2.restart(tr)
    else
        local rs = {
            rollout = opt.rollout,
            dcnn_rollout_per_move = (opt.dcnn_rollout == -1 and opt.rollout or opt.dcnn_rollout),
            rollout_per_move = opt.rollout
        }

        tr = playoutv2.new(rs)
    end
    count = 0
    signature = utils.get_signature()
    io.stderr:write("New MCTS game, signature: " .. signature)
    os.execute("mkdir -p " .. paths.concat(opt.pipe_path, signature))
    playoutv2.print_params(tr)
end

function callbacks.quit_func()
    if tr then
        playoutv2.free(tr)
    end
end

function callbacks.move_predictor(b)
    local prefix = prepare_prefix(opt)
    local m = playoutv2.play_rollout(tr, prefix, b)
    if prefix then io.stderr:write("Save tree to " .. prefix) end
    return m.x + 1, m.y + 1, m.win_rate
end

function callbacks.move_receiver(x, y, player)
    local prefix = prepare_prefix(opt)
    playoutv2.prune_xy(tr, x, y, player, prefix)
end

function callbacks.thread_switch(arg)
    if arg == "on" then
        playoutv2.thread_on(tr)
    elseif arg == 'off' then
        playoutv2.thread_off(tr)
    else
        io.stderr:write("Command " .. arg .. " is not recognized!")
    end
end

local Trainer = require 'rl_network.trainer'
local model = torch.load(opt.model_filename).net
trainer = Trainer(mode, opt, callbacks)
trainer:train(model, opt, callbacks)
