-- @Author: gigaflw
-- @Date:   2017-11-22 15:35:40
-- @Last Modified by:   gigaflw
-- @Last Modified time: 2017-11-23 13:57:20

require 'optim'

local resnet = require 'resnet.resnet'
local get_dataloader = require 'resnet.dataloader'

get_shape_str = function (obj) return table.concat((#obj):totable(), ' x ') end

local meta = {}
local Trainer = torch.class('resnet.Trainer', meta)

function Trainer:__init(net, crit, optim, optim_config)
    self.net = net
    self.crit = crit
    self.optim = optim
    self.optim_config = optim_config
    self.ALL_PARAM, self.ALL_PARAM_GRAD = net:getParameters()
    print(string.format("The network has %s trainable parameters", get_shape_str(self.ALL_PARAM)))
end

function Trainer:train(dataloader, max_batches)
    -- not shuffle yet
    local total_loss, batches = 0, 0

    for ind, inputs, labels in dataloader.iter(max_batches) do
        labels = {labels.a, labels.z} -- array is needed for training
        
        batch_loss = self:step(inputs, labels)[1]
        
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

        return loss, self.ALL_PARAM_GRAD -- ALL_PARAM_GRAD is d(loss)/d(all parameters)
    end  

    local _, batch_loss = self.optim(eval, self.ALL_PARAM, self.optim_config)
    -- losses is a table containing value of the loss function  
    -- (just 1 value for the SGD optimization)  
    return batch_loss
end

local opt = {
    data_augmentation = false,
    batch_size = 24,
    max_batches = 20
}

local sgd_config = {
  learningRate = 1e-2,  
  learningRateDecay = 1e-4,  
  weightDecay = 1e-3,  
  momentum = 1e-4  
} -- these key names should not be changed


net = resnet.create_model()
crit = resnet.create_criterion()

dataloader = get_dataloader('test', opt.batch_size)
dataloader.load_game(1)

trainer = meta.Trainer(net, crit, optim.sgd, sgd_config)
trainer:train(dataloader, opt.max_batches)
