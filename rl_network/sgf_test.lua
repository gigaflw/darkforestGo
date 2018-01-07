--
-- Created by HgS_1217_
-- Date: 2018/1/5
--

package.path = package.path .. ';../?.lua'

local get_dataloader = require 'resnet.dataloader'

local opt = {
    batch_size = 10,
    style = 'traverse',
    data_augment = false,
    data_pool_size = -1,
    dropout = 0.0,
    min_ply = 0,
    no_tie = true,
    verbose = true,
    debug = true,
    no_pass = true,
    n_feature = 25,
}
local dataloader = get_dataloader('../dataset/sina', opt)
for ind, inputs, labels in dataloader.iter(100000) do
--    print(ind)
--    print(inputs)
--    print(labels)
end
