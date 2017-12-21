-- @Author: gigaflw
-- @Date:   2017-12-19 19:50:51
-- @Last Modified by:   gigaflw
-- @Last Modified time: 2017-12-21 09:53:50

local pl = require 'pl.import_into'()
local utils = require 'utils.utils'
local common = require 'common.common'

local RLPlayer = require 'rl_network.player'

local class = require 'class'
local Trainer = class('rl_network.Trainer')

function Trainer:__init(net, opt, callbacks)
    self.opt = {}
    for k, v in pairs(opt) do self.opt[k] = v end 

    self.net = net
    self.player = RLPlayer(callbacks, opt)
    self.resnet = require 'resnet.rl_train'
    self._epoch = 1
end

function Trainer:generate()
    local opt = self.opt
    
    if opt.sgf_dir ~= '' then os.execute('mkdir -p '..paths.concat(opt.sgf_dir, opt.dataset_name)) end
    function save_sgf(sgf)
        if opt.sgf_dir == '' then return end

        local footprint = string.format("%s-%s-%s__%d", utils.get_signature(), self._epoch, utils.get_randString(6), self.player.b._ply)
        local filename = paths.concat(opt.sgf_dir, opt.dataset_name, footprint..'.sgf')
        local f = io.open(filename, "w")
        if not f then return false, "file " .. filename .. " cannot be opened" end
        f:write(sgf)
        f:close()
    end

    self:log('Generating self-play games...')

    local timer = torch.Timer()

    local sgf_dataset = {}
    for g = 1, opt.games do
        self:log(string.format("Generating dataset: %d/%d", g, opt.games))
        local sgf = self:play_one_game()

        if sgf then
            sgf_dataset[g] = sgf
            save_sgf(sgf)
            self:log(string.format("Dataset %d generated in %.4fs", g, timer:time().real))
        else
            print("Unknown error happens, sgf is nil!")
        end
        timer:reset()
    end
    
    local dataset_path = paths.concat(opt.dataset_dir, opt.dataset_name)
    self.resnet.save_sgf_to_dataset(sgf_dataset, dataset_path)  -- generate .bin & .idx file
    self:log(string.format("Dataset '%s.bin' saved", dataset_path))
end

function Trainer:train()
    local opt = self.opt
    local res_opt = self.resnet.get_opt({
        log_file = opt.log_file, -- save to the same log file
        dataset_dir = opt.dataset_dir
    }) 

    self:log('Reinforcement training starts')

    local timer = torch.Timer()

    while self._epoch < opt.epochs do
        local e = self._epoch
        ----------------------------
        -- supervised training
        ----------------------------
        self.resnet.train_on_the_fly(self.net, opt.dataset_name, string.format("rl%04d", e), res_opt)
        self.net = self:get_current_best_model()

        ----------------------------
        -- print result & save ckpt
        ----------------------------
        self:log(string.format("| RL Epoch %d ends in %.4fs", e, timer:time().real))
        timer:reset()

        self.net:clearState()
        if math.fmod(e, opt.epoch_per_ckpt) == 0 then
            self:save(e, string.format('rl.e%04d.params', e))
        end
        self:save(e, 'rl.latest.params') -- save 'latest.params' every epoch

        self._epoch = e + 1
    end

    self:log("Reinforcement training ends")
end

function Trainer:get_current_best_model()
    -- TODO
    -- decide the stronger model between the old and the new
    -- the stronger will be passed to the next epoch
    -- local play_opt, opt1, opt2 = rl_utils.train_play_init(old_model, opt.model,
    --     string.format("resnet_rl%04d", epoch - 1), string.format("resnet_rl%04d", epoch))

    -- local old_win, new_win, differential = self_play.play(opt1, opt2, play_opt)

    -- print(string.format('old_win = %d, new_win = %d, differential = %d', old_win, new_win, differential))

    -- if differential > 0 then
    --     opt.model = old_model  -- TODO: Save or give up the new model
    -- end
    return self.net
end

function Trainer:play_one_game()
    self.player:clear_board()
    local sgf
    while true do
        local valid, move, res = self.player:g()
        if not valid then
            break
        elseif move == "resign" then
            sgf = self:get_sgf_string(res)
            break
        end
    end
    return sgf
end

function Trainer:save(epoch, filename)
    local obj = {
        epoch = epoch,
        net = self.net,
        opt = self.opt,
    }
    filename = self.opt.ckpt_prefix..filename
    torch.save(paths.concat(self.opt.ckpt_dir, filename), obj)
    self:log("checkpoint '"..filename.."' saved")
end

function Trainer:log(message)
    print(message)
    if self.opt.log_file ~= '' then
        local f = io.open(self.opt.log_file, 'a')
        f:write(message..'\n')
        f:close()
    end
end

function Trainer:quit()
    self.player:quit()
end

function Trainer:get_sgf_string(result)
    local ret_str
    if result.resign_side == common.white then
        ret_str = "B+Resign"
    elseif result.resign_side == common.black then
        ret_str = "W+Resign"
    else
        ret_str = result.score > 0 and string.format("B+%.1f", result.score) or string.format("W+%.1f", -result.score)
    end

    local model_name = string.format("resnet%04d", self._epoch)

    return self.player:save_sgf('', ret_str, model_name, model_name, false) -- false means do not save here
end

return Trainer
