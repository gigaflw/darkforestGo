--
-- Created by HgS_1217_
-- Date: 2017/11/27
-- Last Modified by:   gigaflw
-- Last Modified time: 2017-12-16 10:07:58
--

require '_torch_class_patch'

local pl = require 'pl.import_into'()
local ffi = require 'ffi'
local threads = require 'threads'
threads.serialization('threads.sharedserialize')

local common = require 'common.common'
local utils = require 'utils.utils'
local symbols, _ = utils.ffi_include(paths.concat(common.script_path(), "../local_evaluator/cnn_local_exchanger.h"))
local C = ffi.load(paths.concat(common.script_path(), "../libs/liblocalexchanger.so"))

local board = require 'board.board'
local evaluator_util = require 'rl_network.resnet_evaluator_util'
local resnet_util = require 'resnet.util'

local opt = pl.lapp[[
    --pipe_path     (default ".")
    --num_attempt   (default 10)                 number of attempt before wait_board gave up and return nil.
    --max_batch     (default 32)

    ** GPU Options  **
    -g,--gpu             (default 1)        which core to use on a multicore GPU environment
    --use_gpu            (default true)     No use when there is no gpu devices
]]

opt.use_gpu = opt.use_gpu and resnet_util.have_gpu() -- only use gpu when there is one
if opt.use_gpu then
    require 'cutorch'
    cutorch.setDevice(opt.gpu)
    print('use gpu device '..opt.gpu)
end


local SIG_OK = tonumber(symbols.SIG_OK)
local NUM_POSSIBLE_MOVES = 362  -- 19*19 + pass move
local model_filename = 'resnet.ckpt/latest.cpu.params'

---- Loading Model ----
print("Loading model = " .. model_filename)
local model = torch.load(model_filename).net
print("Loading complete")

---- Init Pipe File ----

local ex = C.ExLocalInit(opt.pipe_path, opt.gpu - 1, common.TRUE)
print("CNN Exchanger initialized.")
print("Size of MBoard: " .. ffi.sizeof('MBoard'))
print("Size of MMove: " .. ffi.sizeof('MMove'))
board.print_info()

---- reuseable tensors ---
local block_ids = torch.DoubleTensor(opt.max_batch)
local sorted_prob = torch.FloatTensor(opt.max_batch, NUM_POSSIBLE_MOVES)
local sorted_index = torch.FloatTensor(opt.max_batch, NUM_POSSIBLE_MOVES)
local probs_cuda, sorted_prob_cuda, sorted_index_cuda
if opt.use_gpu then
    probs_cuda = torch.CudaTensor(opt.max_batch, NUM_POSSIBLE_MOVES)
    sorted_prob_cuda = torch.CudaTensor(opt.max_batch, NUM_POSSIBLE_MOVES)
    sorted_index_cuda = torch.CudaLongTensor(opt.max_batch, NUM_POSSIBLE_MOVES)
else
    probs_cuda = torch.FloatTensor(opt.max_batch, NUM_POSSIBLE_MOVES)
    sorted_prob_cuda = torch.FloatTensor(opt.max_batch, NUM_POSSIBLE_MOVES)
    sorted_index_cuda = torch.LongTensor(opt.max_batch, NUM_POSSIBLE_MOVES)
end

evaluator_util.init(opt.max_batch)

print("ready")
io.flush()

local boards = {}
while true do
    -- Get data
    block_ids:zero()
    boards = {}

    local num_valid = 0

    for i = 1, opt.max_batch do
        local mboard = evaluator_util.boards[i - 1]
        local ret = C.ExLocalServerGetBoard(ex, mboard, opt.num_attempt)
        if ret == SIG_OK and mboard.seq ~= 0 and mboard.b ~= 0 then
            num_valid = num_valid + 1
            boards[num_valid] = mboard.board
            block_ids[num_valid] = i
        end
    end

    if C.ExLocalServerIsRestarting(ex) == common.FALSE and probs_cuda ~= nil and num_valid > 0 then
        print(string.format("Valid sample = %d / %d", num_valid, opt.max_batch))
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
