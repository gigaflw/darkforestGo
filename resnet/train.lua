-- @Author: gigaflw
-- @Date:   2017-11-22 15:35:40
-- @Last Modified by:   gigaflw
-- @Last Modified time: 2017-11-22 15:39:21

local make_resnet = require 'resnet.resnet'
local get_dataloader = require 'resnet.dataloader'

local opt = {
    data_augmentation = false,
    batch_size = 24
}

resnet = make_resnet()

dataloader = get_dataloader('test', opt.batch_size)
dataloader.load_random_game()

for ind, input, label in dataloader.iter() do
    p, v = table.unpack(resnet:forward(input))
    -- print(p)
    print(v)
    break
end


