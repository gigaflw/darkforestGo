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
local evaluator_util = require 'rl_network.resnet_evaluator_util'
local threads = require 'threads'
local util = require 'resnet.util'
local pl = require 'pl.import_into'()
local resnet_util = require("resnet.util")

local opt = pl.lapp[[
    --async                                    Make it asynchronized.
    --pipe_path (default "../../dflog")        Path for pipe file. Default is in the current directory, i.e., go/mcts
    --codename  (default "resnet_16")         Code name for the model to load.

    ** GPU Options  **
    -g,--gpu             (default 1)        which core to use on a multicore GPU environment
    --use_gpu            (default true)     No use when there is no gpu devices
]]

opt_evaluator = opt

opt.use_gpu = opt.use_gpu and util.have_gpu() -- only use gpu when there is one

if opt.use_gpu then
    require 'cutorch'
    cutorch.setDevice(opt.gpu)
    print('use gpu device '..opt.gpu)
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

print("Loading model = " .. model_filename)
local model = torch.load(model_filename).net
print("Loading complete")

-- Server side.
local ex = C.ExLocalInit(opt.pipe_path, opt.gpu - 1, common.TRUE)
print("CNN Exchanger initialized.")
print("Size of MBoard: " .. ffi.sizeof('MBoard'))
print("Size of MMove: " .. ffi.sizeof('MMove'))
board.print_info()

local block_ids = torch.DoubleTensor(max_batch)
local sorted_prob = torch.FloatTensor(max_batch, common.board_size * common.board_size + 1)
local sorted_index = torch.FloatTensor(max_batch, common.board_size * common.board_size + 1)

evaluator_util.init(max_batch)

print("ready")
io.flush()

-- Preallocate the cuda tensors.
local probs_cuda, sorted_prob_cuda, sorted_index_cuda
local boards

while true do
    -- Get data
    block_ids:zero()
    boards = {}

    local num_valid = 0

    for i = 1, max_batch do
        local mboard = evaluator_util.boards[i - 1]
        local ret = C.ExLocalServerGetBoard(ex, mboard, num_attempt)
        if ret == sig_ok and mboard.seq ~= 0 and mboard.b ~= 0 then
            if probs_cuda == nil then
                probs_cuda = torch.CudaTensor(max_batch, total_index_length)
                sorted_prob_cuda = torch.CudaTensor(max_batch, total_index_length)
                sorted_index_cuda = torch.CudaLongTensor(max_batch, total_index_length)
            end
            num_valid = num_valid + 1
            boards[num_valid] = mboard.board
            block_ids[num_valid] = i
        end
    end

    if C.ExLocalServerIsRestarting(ex) == common.FALSE and probs_cuda ~= nil and num_valid > 0 then
        print(string.format("Valid sample = %d / %d", num_valid, max_batch))
        local start = common.wallclock()

        local probs_cuda_part = probs_cuda:sub(1, num_valid)
        local sorted_prob_cuda_part = sorted_prob_cuda:sub(1, num_valid)
        local sorted_index_cuda_part = sorted_index_cuda:sub(1, num_valid)

        for k = 1, num_valid do
            local output = resnet_util.play(model, boards[k], boards[k]._ply)
            local probs, win_rate = output[1], output[2]

            probs_cuda_part[k] = probs
            evaluator_util.t_received[block_ids[k]] = common.wallclock()
        end

        torch.sort(sorted_prob_cuda_part, sorted_index_cuda_part, probs_cuda_part, 2, true)

        sorted_prob:sub(1, num_valid):copy(sorted_prob_cuda_part)
        sorted_index:sub(1, num_valid):copy(sorted_index_cuda_part)

        print(string.format("Computation = %f", common.wallclock() - start))

        local start = common.wallclock()

        for k = 1, num_valid do
            local mmove = evaluator_util.prepare_move(block_ids[k], sorted_prob[k], sorted_index[k])
            C.ExLocalServerSendMove(ex, mmove)
        end
        print(string.format("Send back = %f", common.wallclock() - start))
    end

    evaluator_util.sparse_gc()

    if C.ExLocalServerSendAckIfNecessary(ex) == common.TRUE then
        print("Ack signal sent!")
    end
end

evaluator_util.free()
C.ExLocalDestroy(ex)
