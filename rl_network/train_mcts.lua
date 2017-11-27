--
-- Created by HgS_1217_
-- Date: 2017/11/27
--

package.path = package.path .. ';../?.lua'

local CNNPlayerV2 = require 'cnnPlayerV2.cnnPlayerV2Framework'
local pl = require 'pl.import_into'()
local self_play_mcts = require("rl_network.self_play_mcts")

local opt = pl.lapp[[
    --rollout         (default 1000)         The number of rollout we use.
    --dcnn_rollout    (default -1)           The number of dcnn rollout we use (If we set to -1, then it is the same as rollout), if cpu_only is set, then dcnn_rollout is not used.
    --dp_max_depth    (default 10000)        The max_depth of default policy.
    -v,--verbose      (default 1)            The verbose level (1 = critical, 2 = info, 3 = debug)
    --print_tree                             Whether print the search tree.
    --max_send_attempts (default 3)          #attempts to send to the server.
    --pipe_path         (default "../../dflog") Pipe path
    --tier_name         (default "ai.go-evaluator") Tier name
    --server_type       (default "local")    We can choose "local" or "cluster". For open source version, for now "cluster" is not usable.
    --tree_to_json                           Whether we save the tree to json file for visualization. Note that pipe_path will be used.
    --num_tree_thread   (default 16)         The number of threads used to expand MCTS tree.
    --num_gpu           (default 1)          The number of gpus to use for local play.
    --sigma             (default 0.05)       Sigma used to perturb the win rate in MCTS search.
    --use_sigma_over_n                       use sigma / n (or sqrt(nparent/n)). This makes sigma small for nodes with confident win rate estimation.
    --num_virtual_games (default 0)          Number of virtual games we use.
    --acc_prob_thres    (default 0.8)        Accumulated probability threshold. We remove the remove if by the time we see it, the accumulated prob is greater than this thres.
    --max_num_move      (default 20)          Maximum number of moves to consider in each tree node.
    --min_num_move      (default 1)          Minimum number of moves to consider in each tree node.
    --decision_mixture_ratio (default 5.0)   Mixture MCTS count ratio with cnn_confidence.
    --time_limit        (default 1)        Limit time for each move in second. If set to 0, then there is no time limit.
    --win_rate_thres    (default 0.0)        If the win rate is lower than that, resign.
    --use_pondering                          Whether we use pondering
    --exec              (default "")         Whether we run an initial script
    --setup_board       (default "")         Setup board. The argument is "sgfname moveto"
    --dynkomi_factor    (default 0.0)        Use dynkomi_factor
    --num_playout_per_rollout (default 1)    Number of playouts per rollouts.
    --single_move_return                     Use single move return (When we only have one choice, return the move immediately)
    --expand_search_endgame                  Whether we expand the search in end game.
    --default_policy    (default "v2")       The default policy used. Could be "simple", "v2".
    --default_policy_pattern_file (default "../models/playout-model.bin") The patter file
    --default_policy_temperature  (default 0.125)   The temperature we use for sampling.
    --online_model_alpha         (default 0.0)      Whether we use online model and its alpha
    --online_prior_mixture_ratio (default 0.0)      Online prior mixture ratio.
    --use_rave                               Whether we use RAVE.
    --use_cnn_final_score                    Whether we use CNN final score.
    --min_ply_to_use_cnn_final_score (default 100)     When to use cnn final score.
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
    --save_sgf_per_move                               If so, then we save sgf file for each move
    --use_formal_params                               If so, then use formal parameters
    --use_custom_params                               If so, then use custom parameters
]]

local opt2 = {
    rule = opt.rule,
    win_rate_thres = opt.win_rate_thres,
    exec = opt.exec,
    setup_board = opt.setup_board,
    default_policy = opt.default_policy,
    default_policy_pattern_file = opt.default_policy_pattern_file,
    default_policy_temperature = opt.default_policy_temperature,
    default_policy_sample_topn = opt.sample_topn,
    save_sgf_per_move = opt.save_sgf_per_move
}

--self_play_mcts.train(callbacks, opt2)

--local cnnplayer = CNNPlayerV2("CNNPlayerV2MCTS", "go_player_v2_mcts", "1.0", callbacks, opt2)
--cnnplayer:mainloop()


