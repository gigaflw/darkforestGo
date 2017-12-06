--
-- Created by HgS_1217_
-- Date: 2017/11/29
--

local utils
local resnet_util = require("resnet.util")
local goutils = require("utils.goutils")
local board = require('board.board')
local common = require("common.common")
local pl = require 'pl.import_into'()

local rl_utils = {}

function rl_utils.sample(probs)

--    local cum_prob = torch.cumsum(probs, 1)
--    local needle = torch.rand(1)[1]*cum_prob[cum_prob:size()]
--
--    local index = -1
--    for i = 1, cum_prob:size()[1] do
--        if needle < cum_prob[i] then
--            index = i
--            break
--        end
--    end
--
--    return index

--    print(probs)
    local _, index = torch.max(probs, 1)
--    print(index)
    return index[1]
end

function rl_utils.play_with_cnn(b, board_history, player, net)
    local recent_board = {}

    for i = 1, 8 do
        if i > #board_history then
            recent_board[i] = board.new()
        else
            recent_board[i] = board_history[#board_history + 1 - i]
        end
    end

--    for j=1, 8 do
--        board.show_fancy(recent_board[j], "all_rows_cols")
--    end

    local output = resnet_util.play(net, recent_board, player)
    local probs, win_rate = output[1], output[2]
    local x, y

    local max_iter, iter = 363, 1
    while iter < max_iter do
        local index = rl_utils.sample(probs)

        -- pass move
        if index == 362 then
            x, y = 0, 0
            break
        end

        x, y = goutils.moveIdx2xy(index)

        local check_res, comment = goutils.check_move(b, x, y, player)
        if check_res then
            break
        else
            probs[index] = 0
            iter = iter + 1
        end
    end

    if iter >= max_iter then
        x, y = 0, 0
        print("not find a valid move in ", max_iter, " iterations ..")
    end

    return x, y, win_rate
end

function rl_utils.rl_init(options)
    -- opt.feature_type and opt.userank are necessary for the game to be played.
    local opt = pl.tablex.deepcopy(options)
    opt.sample_step = opt.sample_step or -1
    opt.temperature = opt.temperature or 1
    opt.shuffle_top_n = opt.shuffle_top_n or 1
    opt.rank = opt.rank or '9d'

    if opt.usecpu == nil or opt.usecpu == false then
        utils = require 'utils.utils'
        utils.require_torch()
        utils.require_cutorch()
    else
        g_nnutils_only_cpu = true
        utils = require 'utils.utils'
    end

    opt.userank = true
    assert(opt.shuffle_top_n >= 1)

    opt.attention = { 1, 1, common.board_size, common.board_size }

    local opt1, opt2 = pl.tablex.deepcopy(opt), pl.tablex.deepcopy(opt)

    opt1.input = (opt.codename == "" and opt.input or common.codenames[opt.codename1].model_name)
    opt2.input = (opt.codename == "" and opt.input or common.codenames[opt.codename2].model_name)

    opt1.codename = opt.codename1
    opt2.codename = opt.codename2

    opt1.feature_type = opt.codename == "" and opt.feature_type or common.codenames[opt.codename1].feature_type
    opt2.feature_type = opt.codename == "" and opt.feature_type or common.codenames[opt.codename2].feature_type

    local model_name1 = opt.use_local_model and pl.path.basename(opt1.input) or opt1.input
    local model_name2 = opt.use_local_model and pl.path.basename(opt2.input) or opt2.input

    if opt.verbose then print("Load model 1 " .. model_name1) end
    local model1 = torch.load(model_name1)
    if opt.verbose then print("Load model 1 complete") end

    if opt.verbose then print("Load model 2 " .. model_name2) end
    local model2 = torch.load(model_name2)
    if opt.verbose then print("Load model 2 complete") end

    local preSampleModel1, preSampleModel2
    local preSampleOpt1, preSampleOpt2 = pl.tablex.deepcopy(opt1), pl.tablex.deepcopy(opt2)

    if opt.temperature > 1 then
        if opt.verbose then print("temperature: " , opt.temperature) end
        preSampleModel1 = goutils.getDistillModel(model1, opt.temperature)
        preSampleModel2 = goutils.getDistillModel(model2, opt.temperature)
    else
        if opt.presample_codename1 ~= nil and opt.presample_codename1 ~= false then
            local code = common.codenames[opt.presample_codename1]
            if opt.verbose then print("Load preSampleModel 1 " .. code.model_name) end
            preSampleModel1 = torch.load(code.model_name)
            preSampleOpt1.feature_type = code.feature_typ
        else
            preSampleModel1 = model1
        end

        if opt.presample_codename2 ~= nil and opt.presample_codename2 ~= false then
            local code = common.codenames[opt.presample_codename2]
            if opt.verbose then print("Load preSampleModel 2 " .. code.model_name) end
            preSampleModel2 = torch.load(code.model_name)
            preSampleOpt2.feature_type = code.feature_typ
        else
            preSampleModel2 = model2
        end
    end

    opt1.preSampleModel, opt2.preSampleModel = preSampleModel1, preSampleModel2
    opt1.preSampleOpt, opt2.preSampleOpt = preSampleOpt1, preSampleOpt2
    opt1.model, opt2.model = model1, model2
    if opt.valueModel and opt.valueModel ~= "" then
        opt1.valueModel = torch.load(opt.valueModel)
        opt2.valueModel = torch.load(opt.valueModel)
    end

    if opt.verbose then print("dcnn ready!") end

    return opt1, opt2
end

return rl_utils