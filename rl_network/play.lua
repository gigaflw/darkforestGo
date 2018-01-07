--
-- Created by HgS_1217_
-- Date: 2017/12/16
-- @Last Modified by:   gigaflw
-- @Last Modified time: 2018-01-07 17:22:20
--
local pl = require 'pl.import_into'()

local opt = pl.lapp[[
    --mode                 (default 'gtp')          'gtp' or 'battle'
    --mcts                 If given, use mcts search (local evaluator required),
                            in this case, which model to use is decided by the options in 'evaluator.lua'
                            otherwise, use the raw output of the network

    --model                (default "")             Path name to a model file, no use if '--mcts' is given, in which case the used model is decided by 'evaluator.lua'
    --model_type           (default "df2")          'df2' or 'resnet'
    --at_random                                     Select moves according to probability, in stead of choosing the move with highest prob

    ** Player Options **
    --player_name       (default 'maim')
    --player_verbose
    --win_rate_thres    (default 0.0)           If the win rate is lower than that, resign.
    --resign                                    Whether resign when screwed up.
    --resign_thre       (default 10)            If the opponent wins at least this much in fast rollout, we resign.
    --resign_step       (default 20)            Check resign every this steps

    ** 'battle mode' Options **
    --model2               (default "")             Path name to a model file
    --max_ply              (default 1000)           End game in advance
    --sample_step          (default -1)             If the step of a game is less than the threshold, it is a bad sample.
    --num_games            (default 2)              The number of games to be playe.
    --device               (default 3)              GPU device if there are multiple
    --sgf_dir              (default ".")            Where to save sgf

    ************************ MCTS Options (no use if --mcts is not given) ****************************
    ** PlayoutV2 Options **
    --num_gpu           (default 1)
    -v,--verbose        (default 1)             The verbose level (1 = critical, 2 = info, 3 = debug)

    *** cnn evaluator ***
    --pipe_path         (default "./pipes")     Pipe path
    --server_type       (default "local")       We can choose "local" or "cluster". For open source version, for now "cluster" is not usable.
    --max_send_attempts (default 3)             #attempts to send to the server.
    --acc_prob_thres    (default 0.8)           Accumulated probability threshold. We remove the other moves if by the time we see it, the accumulated prob is greater than this thres.

    *** tree search ***
    --cpu_only                                  Whether we only use fast rollout.
    --num_tree_thread       (default 4)         The number of threads used to expand MCTS tree.
    --rollout               (default 1000)      How many games are played in one search
    --dcnn_rollout          (default -1)        The number of dcnn rollout we use (If we set to -1, then it is the same as rollout), if cpu_only is set, then dcnn_rollout is not used.
    --dynkomi_factor        (default 0.0)       MYSTERIOUS
    --single_move_return                        Use single move return (When we only have one choice, return the move immediately)
    --expand_search_endgame                     MYSTERIOUS
    --percent_playout_in_expansion   (default 0)      The percent of threads that will run playout when we expand the node. Other threads will block wait.

    *** default policy ***
    --num_playout_per_rollout (default 1)       Average multiple times of dp playout to get average score for bp
    --dp_max_depth      (default 10000)         The max_depth of default policy, ignored if patternv2 is used
    --default_policy    (default "v2")          The default policy used. Could be "simple", "v2".
    --default_policy_pattern_file (default "models/playout-model.bin") The patter file
    --default_policy_temperature  (default 0.125)   The temperature we use for sampling.
    --default_policy_sample_topn  (default -1)            If use v2, topn we should sample..

    *** backpropagation ***
    --use_cnn_final_score                             Whether we use CNN final score.
    --min_ply_to_use_cnn_final_score (default  60)    When to use cnn final score.
    --final_mixture_ratio            (default 0.5)    The mixture ratio we used.

    *** child node expansion ***
    --expand_n_thres    (default 0)             Statistics collected before expand.
    --use_async                                 Open async model
    --max_num_move      (default 20)            Maximum number of moves to consider in each tree node.
    --min_num_move      (default 1)             Minimum number of moves to consider in each tree node.

    *** best child node ***
    --sigma             (default 0.05)          Sigma used to perturb the win rate in MCTS search.
    --use_old_uct                               old uct has smaller factor for prior
    --use_sigma_over_n                          use sigma / n (or sqrt(nparent/n)). This makes sigma small for nodes with confident win rate estimation.
    --num_virtual_games (default 0)             Seems no use. If > 0, there will be no noise for uct prior
    --decision_mixture_ratio (default 5.0)      score = this_ratio * cnn_pred / factor  + win_rate
    --use_rave                                  Whether we use RAVE.

    *** time control ***
    --time_limit                (default 1)     Limit time for each move in second. If set to 0, then there is no time limit.
    --use_pondering                             MYSTERIOUS
    --heuristic_tm_total_time   (default 0)     Time for heuristic tm (0 mean you don't use it).

    *** online model ***
    --online_model_alpha         (default 0.0)      Whether we use online model and its alpha
    --online_prior_mixture_ratio (default 0.0)      Online prior mixture ratio.

    *** misc ***
    --print_tree                                        Whether print the search tree.
    --tier_name         (default "ai.go-evaluator")     Tier name
    --tree_to_json                                      Whether we save the tree to json file for visualization. Note that pipe_path will be used.
    --rule              (default jp)                    Use JP rule : jp, use CN rule: cn
    --use_formal_params                                 If so, then use formal parameters
    --use_custom_params                                 If so, then use custom parameters
    --min_rollout_peekable           (default 20000)    The command peek will return if the minimal number of rollouts exceed this threshold

    **************************** MCTS Options End ***************************
]]

if pl.path.exists("/dev/nvidiactl") then
    require 'cunn'
    require 'cutorch'
    cutorch.setDevice(opt.device)
else
    require 'nn'
end

-- init callbacks
local callbacks
if opt.mcts then
    local mcts = require 'rl_network.mcts'
    mcts.init(opt)
    callbacks = mcts.callbacks
else
    local model = torch.load(opt.model).net
    local rl_utils = require 'rl_network.utils'

    callbacks = {
        move_predictor = function (board, player)
            assert(player == board._next_player)
            x, y, win_rate = rl_utils.play_with_cnn(board, player, model, opt.at_random, opt.model_type)
            return x, y, win_rate
        end
    }
end
-- init callbacks ends

if opt.mode == 'gtp' then
    local Player = require 'rl_network.player'
    local player = Player(callbacks, opt)
    player:mainloop()
elseif opt.mode == 'battle' then
    local rl_utils = require 'rl_network.utils'
    local self_play = require 'rl_network.self_play'

    -- TODO: self play with mcts
    opt.model1 = opt.model          -- legacy code compatibility
    local dcnn_opt1, dcnn_opt2 = rl_utils.play_init(opt)
    local win1, win2, score = self_play.play(dcnn_opt1, dcnn_opt2, opt)
    print(string.format("model1 wins %.2f%%, %.2f on average", win1/opt.num_games * 100, score / opt.num_games))
    self_play.free()
end
