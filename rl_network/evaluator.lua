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
local resnet_utils = require 'resnet.utils'
local pkg = require 'rl_network.move_package'

local opt = pl.lapp[[
    --pipe_path     (default ".")
    --num_attempt   (default 10)            Number of attempt before wait_board gave up and return nil.
    --max_batch     (default 32)
    --use_dp        (default false)         Whether add to candidate move with default policy

    ** GPU Options  **
    --device             (default 1)        which core to use on a multicore GPU environment
    --use_gpu            (default true)     No use when there is no gpu devices
]]


opt.use_gpu = opt.use_gpu and resnet_utils.have_gpu() -- only use gpu when there is one
if opt.use_gpu then
    require 'cutorch'
    cutorch.setDevice(opt.device)
    print('use gpu device '..opt.device)
else
    opt.device = 1
end

local SIG_OK = tonumber(symbols.SIG_OK)
local NUM_POSSIBLE_MOVES = 362  -- 19*19 + pass move
local TIME_RELOAD = 5  -- (second) the interval to reload the model

local model_filename = 'resnet.ckpt/latest.cpu.params'
-- local rl_model_name = 'resnet.ckpt/16/latest.params'  -- TODO: need to change the rl model name

---- Loading Model ----
print("Loading model = " .. model_filename)
local model = torch.load(model_filename).net
print("Loading complete")

---- Init Pipe File ----

local exchanger = C.ExLocalInit(opt.pipe_path, opt.device - 1, common.TRUE)
print("CNN Exchanger initialized.")
print("Size of MBoard: " .. ffi.sizeof('MBoard'))
print("Size of MMove: " .. ffi.sizeof('MMove'))
board.print_info()

---- reuseable tensors ---
local block_ids = torch.DoubleTensor(opt.max_batch)
local pre_probs, pre_values, pre_probs_sorted_v, pre_probs_sorted_k
if opt.use_gpu then
    pre_probs = torch.CudaTensor(opt.max_batch, NUM_POSSIBLE_MOVES)
    pre_values = torch.CudaTensor(opt.max_batch)
    pre_probs_sorted_v = torch.CudaTensor(opt.max_batch, NUM_POSSIBLE_MOVES)
    pre_probs_sorted_k = torch.CudaLongTensor(opt.max_batch, NUM_POSSIBLE_MOVES)
else
    pre_probs = torch.FloatTensor(opt.max_batch, NUM_POSSIBLE_MOVES)
    pre_values = torch.FloatTensor(opt.max_batch)
    pre_probs_sorted_v = torch.FloatTensor(opt.max_batch, NUM_POSSIBLE_MOVES)
    pre_probs_sorted_k = torch.LongTensor(opt.max_batch, NUM_POSSIBLE_MOVES)
end

pkg.init(opt.max_batch)

print("ready")
io.flush()
utils.dbg_set()

local boards = {}
local last_update_time, current_time = common.wallclock(), nil

-------------------
--   Main Loop   --
-------------------
while true do
    block_ids:zero()      -- block_ids[i] is the index for the i-th valid board received
    boards = {}

    current_time = common.wallclock()
    if current_time - last_update_time > TIME_RELOAD then
        -- last_update_time = current_time
        -- if io.open(rl_model_name, 'r') then
        --     print("Reloading model = " .. rl_model_name)
        --     model = torch.load(rl_model_name).net
        --     print("Reloading complete")
        -- end
    end

    local num_valid = 0   -- we can receive `max_batch` boards simultaneously, but only `num_valid` of then are given

    for i = 1, opt.max_batch do
        local mboard = pkg.boards[i - 1]
        local ret = C.ExLocalServerGetBoard(exchanger, mboard, opt.num_attempt)
        if ret == SIG_OK and mboard.seq ~= 0 and mboard.b ~= 0 then
            num_valid = num_valid + 1
            boards[num_valid] = mboard.board
            block_ids[num_valid] = i
        end
    end

    if C.ExLocalServerIsRestarting(exchanger) == common.FALSE and num_valid > 0 then
        print(string.format("Valid sample = %d / %d", num_valid, opt.max_batch))
        local start = common.wallclock()

        local probs = pre_probs:sub(1, num_valid)
        local values = pre_values:sub(1, num_valid)
        local probs_sorted_v = pre_probs_sorted_v:sub(1, num_valid)
        local probs_sorted_k = pre_probs_sorted_k:sub(1, num_valid)

        for i = 1, num_valid do
            local output = resnet_utils.play(model, boards[i], boards[i]._next_player, true) -- true means no pass

            probs[i] = output[1]  -- a 362-d probability distribution
            values[i] = output[2] * 361 -- a float in [-1, 1], output of value network
            pkg.t_received[block_ids[i]] = common.wallclock()
        end

        -- find the top k moves
        torch.sort(probs_sorted_v, probs_sorted_k, probs, 2, true)

        print(string.format("Computation = %f", common.wallclock() - start))

        local start = common.wallclock()
        for i = 1, num_valid do
            local mmove = pkg.prepare_move(block_ids[i], probs_sorted_v[i], probs_sorted_k[i], values[i], opt.use_dp)
            C.ExLocalServerSendMove(exchanger, mmove)
        end
        print(string.format("Send back = %f", common.wallclock() - start))
    end

    pkg.sparse_gc()

    if C.ExLocalServerSendAckIfNecessary(exchanger) == common.TRUE then
        print("Ack signal sent!")
    end
end

pkg.free()
C.ExLocalDestroy(exchanger)
