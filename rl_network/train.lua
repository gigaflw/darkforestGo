--
-- Created by HgS_1217_
-- Date: 2017/11/27
--

require '_torch_class_patch'

local mcts = require 'rl_network.mcts'
local common = require 'common.common'
local utils = require 'utils.utils'
local board = require 'board.board'
local pl = require 'pl.import_into'()

local opt = pl.lapp[[
    ** Trainer Options **
    --mode               (default 'train')          'generate' | 'train' | 'hybrid'
    --log_file           (default 'rl.ckpt/log.txt')               If given, log will be saved
    --dataset_dir        (default './dataset')      Where to save dataset
    --dataset_name       (default '')               Name for dataset, timestamp by default, ignored for hybrid mode

    *** for generating ***
    ----- the model used to generate games is set in the options of `evaluator.lua`
    --games              (default 20)               The number of self-play games
    --sgf_dir            (default './dataset/sgf')  Where to save sgf files, (will not save if not given)

    *** for training ***
    --epochs             (default 30)
    --epoch_per_ckpt     (default 1)
    --ckpt_dir           (default './rl.ckpt')      Where to store the checkpoints
    --ckpt_prefix        (default '')               Extra info to be prepended to checkpoint files
    --model              (default './rl.ckpt/initial.params')   The initial model. Ignored if resume_ckpt is given.
    --resume_ckpt        (default '')
    --continue                                      Continue from the last epoch
    --initial_dataset    (default '')               Only useful in hybrid mode, the first dataset will be loaded in stead of generated. Have effect even `continue`

    ** GPU Options **
    --use_gpu            (default true)             No use when there is no gpu devices
    --device             (default 4)                Which core to use on a multicore GPU environment

    ** Player Options **
    --win_rate_thres    (default 0.0)           If the win rate is lower than that, resign.
    --exec              (default "")            NO USE
    --setup_board       (default "")            NO USE.Setup board. The argument is "sgfname moveto"
    --resign            (default true)          Whether support resign in rl_training.
    --resign_thre       (default 10)            If the opponent wins at least this much in fast rollout, we resign.
    --resign_step       (default 20)            Check resign every this steps

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
    --sample_topn       (default -1)            If use v2, topn we should sample..

    *** backpropagation ***
    --use_cnn_final_score                             Whether we use CNN final score.
    --min_ply_to_use_cnn_final_score (default 100)    When to use cnn final score.
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
    --decision_mixture_ratio (default 5.0)      Mixture MCTS count ratio with cnn_confidence.
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
]]

-------------------------
-- set default values
-------------------------
mcts.init(opt)

local resnet_utils = require 'resnet.utils'
opt.use_gpu = opt.use_gpu and resnet_utils.have_gpu() -- only use gpu when there is one

if opt.use_gpu then
    require 'cutorch'
    cutorch.setDevice(opt.device)
    print('use gpu device '..opt.device)
end

if opt.dataset_name == '' then
    local time = os.date("*t")
    opt.dataset_name = string.format("%04d-%02d-%02d_%02d-%02d", time.year, time.month, time.day, time.hour, time.min)
    print("set dataset_name to "..opt.dataset_name)
end

local model
if opt.model ~= '' and opt.resume_ckpt == '' and opt.mode ~= 'generate' then
    model = torch.load(opt.model).net
end
-------------------------
-- set default values ends
-------------------------

local Trainer = require 'rl_network.trainer'

trainer = Trainer(model, opt, mcts.callbacks)
if opt.mode == 'train' then
    trainer:train(false)
elseif opt.mode == 'hybrid' then
    trainer:train(true)
elseif opt.mode == 'generate' then
    trainer:generate()
end

trainer:quit()
