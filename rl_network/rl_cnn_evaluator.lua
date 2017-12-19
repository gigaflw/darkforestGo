--
-- Created by HgS_1217_
-- Date: 2017/11/27
--

package.path = package.path .. ';../?.lua'

require("_torch_class_patch")

local utils = require('utils.utils')
local ffi = require 'ffi'
local common = require("common.common")
local board = require('board.board')
local util_pkg = require 'common.util_package'
local threads = require 'threads'
local util = require 'resnet.util'
local pl = require 'pl.import_into'()

local opt = pl.lapp[[
    --async                                    Make it asynchronized.
    --pipe_path (default "../../dflog")        Path for pipe file. Default is in the current directory, i.e., go/mcts
    --codename  (default "darkfores2")         Code name for the model to load.

    ** GPU Options  **
    --use_gpu            (default true)     No use when there is no gpu devices
    --device             (default 1)        which core to use on a multicore GPU environment
]]

opt_evaluator = opt

utils.require_torch()
utils.require_cutorch()

opt.use_gpu = opt.use_gpu and util.have_gpu() -- only use gpu when there is one

if opt.use_gpu then
    require 'cutorch'
    cutorch.setDevice(opt.device)
    print('use gpu device '..opt.device)
end

threads.serialization('threads.sharedserialize')

local symbols, s = utils.ffi_include("../local_evaluator/cnn_local_exchanger.h")
local C = ffi.load("../libs/liblocalexchanger.so")

local sig_ok = tonumber(symbols.SIG_OK)

-- number of attempt before wait_board gave up and return nil.
-- previously this number is indefinite, i.e., wait until there is a package (which might cause deadlock).
local num_attempt = 10

local max_batch = opt.async and 128 or 32
local model_filename = common.codenames[opt.codename].model_name
local feature_type = common.codenames[opt.codename].feature_type
assert(model_filename, "opt.codename [" .. opt.codename .. "] not found!")

print("Loading model = " .. model_filename)
local model = torch.load(model_filename)
print("Loading complete")

-- Server side.
local ex = C.ExLocalInit(opt.pipe_path, opt.device - 1, common.TRUE)
print("CNN Exchanger initialized.")
print("Size of MBoard: " .. ffi.sizeof('MBoard'))
print("Size of MMove: " .. ffi.sizeof('MMove'))
board.print_info()

-- [board_idx, received time]
local block_ids = torch.DoubleTensor(max_batch)
local sortProb = torch.FloatTensor(max_batch, common.board_size * common.board_size)
local sortInd = torch.FloatTensor(max_batch, common.board_size * common.board_size)

util_pkg.init(max_batch, feature_type)

print("ready")
io.flush()

-- Preallocate the cuda tensors.
local probs_cuda, sortProb_cuda, sortInd_cuda

-- Feature for the batch.
local all_features
while true do
    -- Get data
    block_ids:zero()
    if all_features then
        all_features:zero()
    end

    local num_valid = 0

    -- Start the cycle.
    for i = 1, max_batch do
        local mboard = util_pkg.boards[i - 1]
        local ret = C.ExLocalServerGetBoard(ex, mboard, num_attempt)
        if ret == sig_ok and mboard.seq ~= 0 and mboard.b ~= 0 then
            local feature = util_pkg.extract_board_feature(i)
            if feature ~= nil then
                local nplane, h, w = unpack(feature:size():totable())
                if all_features == nil then
                    all_features = torch.CudaTensor(max_batch, nplane, h, w):zero()
                    probs_cuda = torch.CudaTensor(max_batch, h*w)
                    sortProb_cuda = torch.CudaTensor(max_batch, h*w)
                    sortInd_cuda = torch.CudaLongTensor(max_batch, h*w)
                end
                num_valid = num_valid + 1
                all_features[num_valid]:copy(feature)
                block_ids[num_valid] = i
            end
        end
    end
    -- Now all data are ready, run the model.
    if C.ExLocalServerIsRestarting(ex) == common.FALSE and all_features ~= nil and num_valid > 0 then
        print(string.format("Valid sample = %d / %d", num_valid, max_batch))
        util_pkg.dprint("Start evaluation...")
        local start = common.wallclock()
        local output = model:forward(all_features:sub(1, num_valid))
        local territory
        util_pkg.dprint("End evaluation...")
        -- If the output is multitask, only take the first one.
        if type(output) == 'table' then
            -- Territory
            if #output == 4 then
                territory = output[4]
            end
            output = output[1]
        end

        local probs_cuda_sel = probs_cuda:sub(1, num_valid)
        local sortProb_cuda_sel = sortProb_cuda:sub(1, num_valid)
        local sortInd_cuda_sel = sortInd_cuda:sub(1, num_valid)

        torch.exp(probs_cuda_sel, output:view(num_valid, -1))
        torch.sort(sortProb_cuda_sel, sortInd_cuda_sel, probs_cuda_sel, 2, true)

        sortProb:sub(1, num_valid):copy(sortProb_cuda_sel)
        sortInd:sub(1, num_valid):copy(sortInd_cuda_sel)

        local score
        if territory then
            -- Compute score, only if > 0.6 we regard it as black/white territory.
            local diff = territory[{{}, {1}, {}}] - territory[{{}, {2}, {}}]
            score = diff:ge(0):sum(3)
        end
        print(string.format("Computation = %f", common.wallclock() - start))

        local start = common.wallclock()
        -- Send them back.
        for k = 1, num_valid do
            local mmove = util_pkg.prepare_move(block_ids[k], sortProb[k], sortInd[k], score and score[k])
            util_pkg.dprint("Actually send move")
            C.ExLocalServerSendMove(ex, mmove)
            util_pkg.dprint("After send move")
        end
        print(string.format("Send back = %f", common.wallclock() - start))

    end

    util_pkg.sparse_gc()

    -- Send control message if necessary.
    if C.ExLocalServerSendAckIfNecessary(ex) == common.TRUE then
        print("Ack signal sent!")
    end
end

C.ExLocalDestroy(ex)
