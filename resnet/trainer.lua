-- @Author: gigaflw
-- @Date:   2017-11-22 15:35:40
-- @Last Modified by:   gigaflw
-- @Last Modified time: 2017-11-25 16:15:51

local lfs = require 'lfs'
local class = require 'class'
local Trainer = class('resnet.Trainer')

local default_opt = {
    batch_size = 3,
    max_batches = 4,
    epoches = 3,
    epoch_per_display = 1,
    epoch_per_ckpt = 1,
    ckpt_dir = './resnet.ckpt'
}

function Trainer:__init(net, crit, optim, optim_opt, opt)
    local doc = [[
        @param: net: [ string | torch.nn.Module ]
            if `string`, then the checkpoint with the same name will be loaded
        @param: crit: [ torch.nn.Criterion ]
            the criterion used to calculate loss
        @param: optim:
            the optimizer used to update parameters
        @param: optim_opt:
            the options for optimizer, refer to the doc of the specific optimizer for entries
        @param: opt:
            the options for everything, once set, cannot be modified
    ]]

    self.crit = crit
    self.optim = optim
    self.optim_opt = optim_opt

    self.opt = {})
    setmetatable(self.opt, {
        __index = function(t, key)
            local val = opt[key] or default_opt[key]
            assert(val ~= nil, "Trainer: option '"..key.."' not found!")
            return val
        end
    })

    lfs.mkdir(opt.ckpt_dir)  -- nothing will happen if the dir already exists

    if type(net) == 'string' then
        self:load_params(net)
    else
        self.net = net
    end

    self.all_params, self.all_params_grad = self.net:getParameters()
     -- all_params_grad will store d(loss)/d(all parameters)
    print(string.format("The network has %d trainable parameters", (#self.all_params)[1]))
end


function Trainer:train(dataloader)
    -- not shuffle yet
    opt = self.opt

    for e = 1, opt.epoches do
        local epoch_loss, batches = 0, 0

        for ind, inputs, labels in dataloader.iter(opt.max_batches) do
            labels = {labels.a, labels.z} -- array is needed for training

            batch_loss = self:_step(inputs, labels)[1]
            -- (just 1 value for the SGD optimization)
            -- FIXME: non-sgd optimzier not supported yet

            batches = batches + 1
            epoch_loss = epoch_loss + batch_loss
            print(string.format("Batch %d loss: %4f", batches, batch_loss))
        end

        epoch_loss = epoch_loss / batches
        
        if math.fmod(e, opt.epoch_per_display) == 0 then
            print(string.format("Epoch %d loss: %4f", e, epoch_loss))
        end

        if math.fmod(e, opt.epoch_per_ckpt) == 0 then
            self.net:clearState()
            self:save(string.format('e%04d.params', e))
        end
    end

    if math.fmod(opt.epoches, opt.epoch_per_ckpt) ~= 0 then
        self.net:clearState()
        self:save(string.format('e%04d.params', opt.epoches))
    print("Training ends")
end

function Trainer:_step(inputs, labels)
    local eval = function(new_param)
        -- the argument and return of this function is required by `torch.optim`
        self.net:zeroGradParameters()
        self.net:forward(inputs)

        local loss = self.crit:forward(self.net.output, labels)
        local loss_grad = self.crit:backward(self.net.output, labels)
        self.net:backward(inputs, loss_grad)

        return loss, self.all_params_grad
    end  

    local _, loss = self.optim(eval, self.all_params, self.optim_opt)
    -- loss is a table containing values of the loss function  
    return loss
end

function Trainer:save(filename)
    torch.save(paths.concat(self.opt.ckpt_dir, filename), self.net)
    torch.save(paths.concat(self.opt.ckpt_dir, 'latest.params'), self.net)
    print("checkpoint '"..filename.."' saved")
end

function Trainer:load(filename)
    filename = filename or 'latest.params'
    self.net = torch.load(paths.concat(self.opt.ckpt_dir, filename))
    print("checkpoint '"..filename.."' loaded")
end

return Trainer
