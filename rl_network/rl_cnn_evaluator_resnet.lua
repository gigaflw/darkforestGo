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
local resnet_util = require("resnet.util")

local opt = pl.lapp[[
    --async                                    Make it asynchronized.
    --pipe_path (default "../../dflog")        Path for pipe file. Default is in the current directory, i.e., go/mcts
    --codename  (default "resnet_16")         Code name for the model to load.

    ** GPU Options  **
    --use_gpu            (default true)     No use when there is no gpu devices
    --device             (default 2)        which core to use on a multicore GPU environment
]]

opt_evaluator = opt

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

local total_index_length = 362
local max_batch = opt.async and 128 or 32
local model_filename = common.codenames[opt.codename].model_name
assert(model_filename, "opt.codename [" .. opt.codename .. "] not found!")

print("Loading model = " .. model_filename)
local model = torch.load(model_filename).net
print("Loading complete")

-- Server side.
local ex = C.ExLocalInit(opt.pipe_path, opt.device, common.TRUE)
print("CNN Exchanger initialized.")
print("Size of MBoard: " .. ffi.sizeof('MBoard'))
print("Size of MMove: " .. ffi.sizeof('MMove'))
board.print_info()

-- [board_idx, received time]
local block_ids = torch.DoubleTensor(max_batch)
local sort_prob = torch.FloatTensor(max_batch, common.board_size * common.board_size)
local sort_ind = torch.FloatTensor(max_batch, common.board_size * common.board_size)

util_pkg.init(max_batch, "custom")

print("ready")
io.flush()

-- Preallocate the cuda tensors.
local probs_cuda, sort_prob_cuda, sort_ind_cuda
local boards

while true do
    -- Get data
    block_ids:zero()
    boards = {}

    local num_valid = 0

    -- Start the cycle.
    for i = 1, max_batch do
        local mboard = util_pkg.boards[i - 1]
        local ret = C.ExLocalServerGetBoard(ex, mboard, num_attempt)
        if ret == sig_ok and mboard.seq ~= 0 and mboard.b ~= 0 then
            print("%d %d", i, max_batch)
            if probs_cuda == nil then
                probs_cuda = torch.CudaTensor(max_batch, total_index_length)
                sort_prob_cuda = torch.CudaTensor(max_batch, total_index_length)
                sort_ind_cuda = torch.CudaLongTensor(max_batch, total_index_length)
            end
            num_valid = num_valid + 1
            boards[num_valid] = mboard.b
            block_ids[num_valid] = i
        end
    end
    -- Now all data are ready, run the model.
    if C.ExLocalServerIsRestarting(ex) == common.FALSE and probs_cuda ~= nil and num_valid > 0 then
        print(string.format("Valid sample = %d / %d", num_valid, max_batch))
        util_pkg.dprint("Start evaluation...")
        local start = common.wallclock()

        local probs_cuda_sel = probs_cuda:sub(1, num_valid)
        local sort_prob_cuda_sel = sort_prob_cuda:sub(1, num_valid)
        local sort_ind_cuda_sel = sort_ind_cuda:sub(1, num_valid)
    
        for i = 1, num_valid do
            local b = boards[i]
            local output = resnet_util.play(model, b, b._ply)
            local probs, win_rate = output[1], output[2]
            probs_cuda_sel[i] = probs
        end

        util_pkg.dprint("End evaluation...")

        torch.sort(sort_prob_cuda_sel, sort_ind_cuda_sel, probs_cuda_sel, 2, true)

        sort_prob:sub(1, num_valid):copy(sort_prob_cuda_sel)
        sort_ind:sub(1, num_valid):copy(sort_ind_cuda_sel)

        print(string.format("Computation = %f", common.wallclock() - start))

        local start = common.wallclock()
        -- Send them back.
        for k = 1, num_valid do
            local mmove = util_pkg.prepare_move(block_ids[k], sort_prob[k], sort_ind[k])
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
