-- @Author: gigaflw
-- @Date:   2017-11-22 15:35:40
-- @Last Modified by:   gigaflw
-- @Last Modified time: 2017-11-24 15:29:34

local class = require 'class'
local Trainer = class('resnet.Trainer')

local default_opt = {
    batch_size = 24,
    max_batches = 20,
    epoches = 20,
    epoch_per_display = 1
}

function Trainer:__init(net, crit, optim, optim_opt, opt)
    self.net = net
    self.crit = crit
    self.optim = optim
    self.optim_opt = optim_opt

    self.opt = {}
    setmetatable(self.opt, {
        __index = function(t, key)
            local val = opt[ket] or default_opt[key]
            assert(val ~= nil, "Trainer: option '"..key.."' not found!")
            return val
        end
    })

    self.all_params, self.all_params_grad = net:getParameters()
     -- all_params_grad will store d(loss)/d(all parameters)
    print(string.format("The network has %d trainable parameters", (#self.all_params)[1]))
end


function Trainer:train(dataloader)
    -- not shuffle yet
    
    for e = 1, self.opt.epoches do
        local epoch_loss, batches = 0, 0

        for ind, inputs, labels in dataloader.iter(self.opt.max_batches) do
            labels = {labels.a, labels.z} -- array is needed for training

            batch_loss = self:step(inputs, labels)[1]
            -- (just 1 value for the SGD optimization)
            -- FIXME: non-sgd optimzier not supported yet
            
            batches = batches + 1
            epoch_loss = epoch_loss + batch_loss
            print(string.format("Batch %d loss: %4f", batches, epoch_loss))
        end

        epoch_loss = epoch_loss / batches
        print(string.format("Epoch %d loss: %4f", e, epoch_loss))
    end
    print("Training ends")
end

function Trainer:step(inputs, labels)
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

return Trainer
