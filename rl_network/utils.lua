--
-- Created by HgS_1217_
-- Date: 2017/11/29
--

local resnet_utils = require 'resnet.utils'
local goutils = require 'utils.goutils'
local common = require 'common.common'
local pl = require 'pl.import_into'()

local rl_utils = {}

function rl_utils.play_with_cnn(b, player, net)

    local output = resnet_utils.play(net, b, player)
    local probs, win_rate = output[1], output[2]

    local prob_sorted, index_sorted = torch.sort(probs, 1, true)

    local x, y
    for i = 1, 362 do
        if index_sorted[i] ~= 362 then -- no pass move
            x, y = goutils.moveIdx2xy(index_sorted[i])

            local check_res, comment = goutils.check_move(b, x, y, player)
            if check_res then break else x, y = nil, nil end -- if is legal move
        end
    end

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

    function _set_opt(model_name, opt)
        if common.codenames[model_name] ~= nil then
            opt.input = common.codenames[model_name].model_name
            opt.codename = model_name
            opt.feature_type = model_name == "" and opt.feature_type or common.codenames[model_name].feature_type
            opt.model = torch.load(opt.input)
        else
            opt.input = model_name
            opt.codename = paths.basename(model_name):match('(.+)%..+')
            opt.feature_type = opt.codename:match('df2') and 'extended' or 'custom'
            opt.model = torch.load(opt.input).net
        end
    end

    _set_opt(opt.model1, opt1)
    _set_opt(opt.model2, opt2)

    return opt1, opt2
end

return rl_utils