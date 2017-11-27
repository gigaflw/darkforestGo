-- @Author: gigaflw
-- @Date:   2017-11-22 15:35:40
-- @Last Modified by:   gigaflw
-- @Last Modified time: 2017-11-27 15:44:08

local lfs = require 'lfs'
local class = require 'class'
local Trainer = class('resnet.Trainer')

local default_opt = {
    batch_size = 24,
    max_batches = 20,
    test_batches = 20,
    epochs = 3,
    epoch_per_ckpt = 1,
    ckpt_dir = './resnet.ckpt',

    lr = 0.1,
    lr_decay = 5e-5,
    wd = 1e-4,
    momentum = 0.9
}

function Trainer:__init(net, crit, opt, train_dataloader, test_dataloader)
    local doc = [[
        @param: net: [ string | torch.nn.Module ]
            if `string`, then the checkpoint with the same name will be loaded
        @param: crit: [ torch.nn.Criterion ]
            the criterion used to calculate loss
        @param: opt:
            the options for everything, once set, cannot be modified
        @param: dataloader:
            a dataloader from `resnet.dataloader`, with which one can
            for ind, inputs, labels in dataloader.iter(opt.max_batches) do
                -- inputs: a tensor in the shape of batch_size x feature_plane x 19 x 19
                -- labels.a: a batch_size-d vector indicating the move
                -- labels.z: a batch_size-d 0-1 vector indicating win or lose
            end
            all float tensors
    ]]

    self.crit = crit
    self.optim = (require 'optim').sgd
    self.optim_state = {
        learningRate = opt.lr or default_opt.lr,
        learningRateDecay = opt.lr_decay or default_opt.lr_decay,
        weightDecay = opt.wd or default_opt.wd,
        momentum = opt.momentum or default_opt.momentum,
        nesterov = true,
        dampening = 0.0
    }

    if opt.use_gpu then
        require 'cunn'
        self.inputs = torch.CudaFloatTensor()
        self.labels = {torch.CudaFloatTensor(), torch.CudaFloatTensor()}
    else
        -- size is adpated to the dataset while training/testing
        self.inputs = torch.FloatTensor()
        self.labels = {torch.FloatTensor(), torch.FloatTensor()}
    end

    self.train_dataloader = train_dataloader
    self.test_dataloader = test_dataloader

    self.opt = {}
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

function Trainer:train()
    -- not shuffle yet
    opt = self.opt

    local function _eval()
        -- the argument and return of this function is required by `torch.optim`
        return self.crit.output, self.all_params_grad
    end

    self.net:training() -- set the model to training mode (this will affect dropout and batch normalization)

    print('Training starts')
    local timer = torch.Timer()

    for e = 1, opt.epochs do
        for ind, inputs, labels in self.train_dataloader.iter(opt.max_batches) do
            labels = {labels.a, labels.z} -- array is needed for training, not table
            self:copy_data(inputs, labels) -- move data to gpu if in gpu mode
            local data_time = timer:time().real
            ----------------------------
            -- update parameters
            ----------------------------
            self.net:forward(self.inputs)
            self.crit:forward(self.net.output, self.labels)

            self.net:zeroGradParameters()

            self.crit:backward(self.net.output, self.labels)
            self.net:backward(self.inputs, self.crit.gradInput)

            self.optim(_eval, self.all_params, self.optim_state)

            ----------------------------
            -- print result
            ----------------------------
            local update_time = timer:time().real
            local top1, top5 = self:accuracy(self.net.output, labels)

            print(string.format("| Epoch %d [%02d/%d], data time: %.3fs, time: %.3fs, loss: %4f, top1 acc: %.5f%%, top5 acc: %.5f%%",
                e, ind, opt.max_batches, data_time, update_time - data_time, self.crit.output, top1 * 100, top5 * 100))
            timer:reset()
        end

        ----------------------------
        -- save ckpt & test
        ----------------------------
        if math.fmod(e, opt.epoch_per_ckpt) == 0 then
            self.net:clearState()
            self:save(string.format('e%04d.params', e))
        end
        if math.fmod(e, opt.epoch_per_test) == 0 then
            self.net:clearState()
            self:test()
        end
    end

    if math.fmod(opt.epochs, opt.epoch_per_ckpt) ~= 0 then
        self.net:clearState()
        self:save(string.format('e%04d.params', opt.epochs))
    end
    print("Training ends")
end

function Trainer:test()
    -- not shuffle yet
    opt = self.opt

    local function _eval()
        -- the argument and return of this function is required by `torch.optim`
        return self.crit.output, self.all_params_grad
    end
    self.net:evaluate() -- set the model to non-training mode (this will affect dropout and batch normalization)

    print('Test starts')
    local epoch_loss, top1_sum, top5_sum, batches = 0.0, 0.0, 0.0, 0
    local timer = torch.Timer()

    for ind, inputs, labels in self.test_dataloader.iter(opt.test_batches) do
        self:copy_data(inputs, labels) -- move data to gpu if in gpu mode
        local data_time = timer:time().real

        local batch_size = (#labels.a)[1]
        labels = {labels.a, labels.z} -- array is needed for training

        self.net:forward(self.inputs)
        self.crit:forward(self.net.output, self.labels)

        local compute_time = timer:time().real

        topv, topi = self.net.output[1]:topk(5)
        acc = topi:eq(labels[1]:long():view(-1, 1):expandAs(topi))

        local top1, top5 = self:accuracy(self.net.output, labels)
        top1_sum = top1_sum + top1
        top5_sum = top5_sum + top5
        epoch_loss = epoch_loss + self.crit.output
        batches = batches + 1

        print(string.format("* Test [%d/%d], data time: %.3fs , time: %.3fs, loss: %4f, top1 acc: %.5f%%, top5 acc: %.5f%%",
            ind, opt.test_batches, data_time, compute_time - data_time, self.crit.output, top1 * 100, top5 * 100))
        timer:reset()
    end
    self.net:training()

    print(string.format("* **\n* Tested %d batches, aver loss: %.5f, top1 acc: %.5f%% top5 acc: %.5f%%\n* **",
            batches, epoch_loss / batches, top1_sum / batches * 100, top5_sum / batches * 100))
    print('Test ends')
end

function Trainer:accuracy(outputs, labels)
    local p, v = outputs[1], outputs[2]
    local a, z = labels[1], labels[2]
    local batch_size = (#v)[1]

    topv, topi = p:topk(5)
    acc = topi:eq(a:long():view(-1, 1):expandAs(topi))

    local top1 = acc:narrow(2, 1, 1):sum() / batch_size
    local top5 = acc:narrow(2, 1, 5):sum() / batch_size

    return top1, top5
end

function Trainer:copy_data(inputs, labels)
    -- is resizing time-consuming?
    if #inputs ~= #self.inputs then self.inputs:resize(#inputs) end
    self.inputs:copy(inputs)
    for i, mat in pairs(self.labels) do
        if #labels[i] ~= #mat then mat:resize(#labels[i]) end
        mat:copy(labels[i])
    end
end

function Trainer:save(filename)
    local obj = {
        net = self.net,
        optim_state = self.optim_state
    }
    torch.save(paths.concat(self.opt.ckpt_dir, filename), obj)
    torch.save(paths.concat(self.opt.ckpt_dir, 'latest.params'), obj)
    print("checkpoint '"..filename.."' saved")
end

function Trainer:load(filename)
    filename = filename or 'latest.params'
    local obj = torch.load(paths.concat(self.opt.ckpt_dir, filename))
    self.net = obj.net
    self.optim_state = obj.optim_state
    print("checkpoint '"..filename.."' loaded")
end

return Trainer
