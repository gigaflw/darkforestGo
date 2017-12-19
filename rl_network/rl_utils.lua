--
-- Created by HgS_1217_
-- Date: 2017/11/29
--

local resnet_util = require("resnet.util")
local goutils = require("utils.goutils")
local common = require("common.common")
local pl = require 'pl.import_into'()

local rl_utils = {}

function rl_utils.sample(probs, ply)

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

    local score, index = torch.sort(probs, 1, true)

--    local res = {}
--    for i=1, score:size(1) do
--        local x, y = goutils.moveIdx2xy(index[i])
--        res[i] = tostring(x) .. "\t" .. tostring(y) .. "\t" .. tostring(score[i])
--    end
--
--    print(res)

--    local x, y = goutils.moveIdx2xy(index[1])
--    print(tostring(x) .. "\t" .. tostring(y) .. "\t" .. tostring(score[1]))


--    if index[1] == 362 then print(tostring(ply) .. "\t" .. tostring(score[1])) end

    return index[1]
end

function rl_utils.play_with_cnn(b, player, net)

    local output = resnet_util.play(net, b, player)
    local probs, win_rate = output[1], output[2]
    local x, y, index

    local max_iter, iter = 363, 1
    while iter < max_iter do
        index = rl_utils.sample(probs, b._ply)

        -- pass move
        if index == 362 then
--            x, y = 0, 0
--            break
            probs[index] = 0
            iter = iter + 1
        else
            x, y = goutils.moveIdx2xy(index)

            local check_res, comment = goutils.check_move(b, x, y, player)
            if check_res then
                break
            else
                probs[index] = 0
                iter = iter + 1
            end
        end
    end

    if iter >= max_iter then
        x, y = 0, 0
        print("not find a valid move in ", max_iter, " iterations ..")
    end

--    print(tostring(b._ply) .. "\t" .. tostring(index) .. "\t" .. tostring(probs[index]))

    return x, y, win_rate
end

function rl_utils.rl_init(opt)
    opt.handi = 0
    opt.komi = 7.5

    opt.input = common.codenames[opt.model_name].model_name
    opt.model = torch.load(opt.input).net

    return opt
end

function rl_utils.train_play_init(old_model, new_model, old_model_name, new_model_name)
    local opt = {}
    opt.handi = 0
    opt.komi = 7.5
    opt.num_games = 2
    opt.sample_step = -1
    opt.pipe_path = "../dflog"

    local opt1, opt2 = pl.tablex.deepcopy(opt), pl.tablex.deepcopy(opt)

    opt1.model = old_model
    opt2.model = new_model

    opt1.codename = old_model_name
    opt2.codename = new_model_name

    return opt, opt1, opt2
end

function rl_utils.play_init(opt)
    opt.shuffle_top_n = 300
    opt.rank = '9d'
    opt.handi = 0
    opt.komi = 7.5
    opt.feature_type = 'old'

    opt.userank = true
    opt.attention = { 1, 1, common.board_size, common.board_size }

    local opt1, opt2 = pl.tablex.deepcopy(opt), pl.tablex.deepcopy(opt)

    opt1.input = common.codenames[opt.codename1].model_name
    opt2.input = common.codenames[opt.codename2].model_name

    opt1.codename = opt.codename1
    opt2.codename = opt.codename2

    opt1.feature_type = opt.codename == "" and opt.feature_type or common.codenames[opt.codename1].feature_type
    opt2.feature_type = opt.codename == "" and opt.feature_type or common.codenames[opt.codename2].feature_type

    local model_name1 = opt1.input
    local model_name2 = opt2.input

    local model1 = opt1.codename == "darkfores2" and torch.load(model_name1) or torch.load(model_name1).net
    local model2 = opt2.codename == "darkfores2" and torch.load(model_name2) or torch.load(model_name2).net

    opt1.model, opt2.model = model1, model2

    return opt1, opt2
end

return rl_utils