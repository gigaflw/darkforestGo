-- @Author: gigaflw
-- @Date:   2017-11-22 15:35:40
-- @Last Modified by:   gigaflw
-- @Last Modified time: 2017-11-24 13:08:30

local class = require 'class'
local Trainer = class('resnet.Trainer')

function Trainer:__init(net, crit, optim, optim_config)
    self.net = net
    self.crit = crit
    self.optim = optim
    self.optim_config = optim_config
    self.all_params, self.all_params_grad = net:getParameters()
     -- all_params_grad will store d(loss)/d(all parameters)
    print(string.format("The network has %d trainable parameters", (#self.all_params)[1]))
end

function Trainer:train(dataloader, max_batches)
    -- not shuffle yet
    local total_loss, batches = 0, 0

    for ind, inputs, labels in dataloader.iter(max_batches) do
        labels = {labels.a, labels.z} -- array is needed for training

        batch_loss = self:step(inputs, labels)[1]
        -- (just 1 value for the SGD optimization)
        -- FIXME: non-sgd optimzier not supported yet
        
        batches = batches + 1
        total_loss = total_loss + batch_loss
        print(string.format("Batch %d loss: %4f", batches, batch_loss))
    end
    print("Training ends")
    return total_loss / batches
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

    local _, loss = self.optim(eval, self.all_params, self.optim_config)
    -- loss is a table containing values of the loss function  
    return loss
end

return Trainer
