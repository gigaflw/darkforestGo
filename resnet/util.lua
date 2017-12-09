-- @Author: gigaflw
-- @Date:   2017-11-29 16:25:36
-- @Last Modified by:   gigaflw
-- @Last Modified time: 2017-12-09 17:57:44

local pl = require 'pl.import_into'()

require 'nn'
if pl.path.exists("/dev/nvidiactl") then
    require 'cunn'
    require 'cudnn'
end

local argcheck = require 'argcheck'
local CBoard = require 'board.board'
local common = require 'common.common'
local goutils = require 'utils.goutils'
local ffi = require 'ffi'
local C = ffi.load(paths.concat(common.script_path(), "../libs/libboard.so"))

local util = {}
-----------------------
-- Feature & Label Extraction
-----------------------
local function get_all_sensible_stones(board, player)
    local moves = ffi.new("AllMoves")
    C.FindAllCandidateMoves(board, player, -1, moves)
    -- the third parameter is not used because I have modified the C code to disable self-atari check

    local ret = { }
    for i = 0, moves.num_moves - 1 do
        table.insert(ret, moves.moves[i] + 1) -- lua begin with 1, c return with 0
    end
    return ret
end

util.board_to_features = argcheck{
    doc = [[
        Given a 19x19 board and the current player,
        return 12x19x19 feature planes
        -- 1: our stones
        -- 2: their stones
        -- 3: empty vertexes
        -- 4: our stones with 1 liberty
        -- 5: our stones with 2 liberties
        -- 6: our stones with 3 or more liberties
        -- 7: their stones with 1 liberty
        -- 8: their stones with 2 liberty
        -- 9: their stones with 3 or more liberty
        -- 10: my history
        -- 11: their history
        -- 12: sensible positions (legal & non-self-atari)
    ]],
    {name='board', type='cdata', help='A `board.board` instance'},
    {name='player', type='number', help='Current player, one of common.black ( = 1 ) or common.white ( = 2 )'},
    call = function (board, player)
        F = torch.FloatTensor(12, 19, 19):zero()

        local p = player
        local o = CBoard.opponent(player)

        local p_stones = CBoard.get_stones(board, p)
        local p_liberties = CBoard.get_liberties_map(board, p)
        local p_history = CBoard.get_history(board, p)

        local o_stones = CBoard.get_stones(board, o)
        local o_liberties = CBoard.get_liberties_map(board, o)
        local o_history = CBoard.get_history(board, o)

        F[1] = p_stones
        F[2] = o_stones
        F[3] = CBoard.get_stones(board, common.empty)

        F[4] = p_liberties:eq(1)
        F[5] = p_liberties:eq(2)
        F[6] = p_liberties:ge(3)

        F[7] = o_liberties:eq(1)
        F[8] = o_liberties:eq(2)
        F[9] = o_liberties:ge(3)

        local curr_ply = CBoard.get_ply(board)
        p_history:add(-curr_ply):exp()
        o_history:add(-curr_ply):exp()

        F[10] = p_history
        F[11] = o_history

        local sensible_moves = get_all_sensible_stones(board, player)

        for _, idx in pairs(sensible_moves) do
            local x, y = goutils.moveIdx2xy(idx)
            F[{12, x, y}] = 1
        end

        return F
    end
}

local _old_board_to_features = argcheck{
    doc = [[
        DEPRECTED:
            this function has been replaced with 'resnet_util.board_to_features'
            because this feature used by Zero is too plain for our coarse network

        Given a 19x19 board and the current player
        extracts 17 feature planes
        return a 17 x 19 x 19 tensor.

        Assume our stones at this time to be X_t (19 x 19), opponent's to be Y_t
        Then the feature planes: (according to AlphaGo Zero thesis)
        s_t = [X_t, Y_t, X_{t-1}, Y_{t-1}, ..., X_{t-7}, Y_{t-7}, C]
        where C is all 1 if we are black, 0 if white.

        Since history info is needed, we need `last_features` to passed into this function
        changes will be both made in place and returned.

        usage:
        > a = <some_tensor>
        > new_a = board_to_features(..., a)
        > new_a = a -- true
    ]],
    {name='board', type='cdata', help='A `board.board` instance'},
    {name='player', type='number', help='Current player, one of common.black ( = 1 ) or common.white ( = 2 )'},
    {name='last_features', type='torch.FloatTensor',
        help='Feature from last iteration since history info is needed'},
    call = function (board, player, last_features)
        ret = last_features or torch.FloatTensor(17, 19, 19):zero()

        for i = 16, 4, -2 do
            ret[i] = ret[i-3]
            ret[i-1] = ret[i-2]
        end

        ret[1] = CBoard.get_stones(board, player)
        ret[2] = CBoard.get_stones(board, CBoard.opponent(player))

        ret:narrow(1, 17, 1):fill(player == common.black and 1 or 0)

        return ret
    end
}

-----------------------
-- Play with Model API
-----------------------
function util.play(model, board_history, player)
    doc = [[
        @param: model:
            a network given by `resnet.resnet.create_model' or `torch.load(<ckpt>)`
        @param: board_with_history:
            an array like this: 
                [B_t, B_{t-1}, ..., B_{t-7}]
            
            where, B_t is a `board.board` instance of the current position,
            B_{t-i} is the history position `i` step ahead,
            i.e. after any of the players playing a stone on B_{t-1}, it becomes B_t.
            if there is no such previous position, make it all zero
        @param: player:
            `common.black` or `common.white`
        @return: a table like this:
            { 1: < 362-d vector, a probability for all moves, sum to 1 >, 2: < -1~1, current player's winning rate > }
            where
                vector[362] is the prob for pass
                vector[idx] is the prob for goutils.moveIdx2xy(idx)

        demo:
            local net = torch.load('./resnet.ckpt/latest.params')

            net = net.net
            board_history = {}
            for i = 1, 8 do
                board_history[i] = CBoard.new()
                CBoard.play(board_history[i], math.random(19), math.random(19), math.random(2) == 1 and common.black or common.white)
            end
            out = play(net, board_history, common.black)
    ]]
    -- local input = torch.CudaTensor(1, 17, 19, 19):zero()
    -- for i = 1, 8 do
    --     input[1][2*i-1] = CBoard.get_stones(board_history[i], player)
    --     input[1][2*i] = CBoard.get_stones(board_history[i], CBoard.opponent(player))
    -- end
    -- input:narrow(2, 17, 1):fill(player == common.black and 1 or 0)
    local input = util.board_to_features(board_history[1], player)

    output = model:forward(input:resize(1, table.unpack((#input):totable())):cuda())
    output[1] = nn.SoftMax():forward(output[1]:double())
    return output
end


-----------------------
-- Print things
-----------------------
function util.print_grad(net, file)
    local need_print = { SpatialConvolution='conv', SpatialBatchNormalization='bn', }
    
    local _print = function(message)
        if file then file:write(message) else print(message) end
    end

    local _print_grad
    _print_grad = function(layer)
        local name = layer.__typename:sub(layer.__typename:find('%.') + 1)

        if name == 'Sequential' or name == 'ConcatTable' then
            for i, m in pairs(layer.modules) do _print_grad(m) end
        elseif need_print[name] then
            _print(string.format(
                "\t%s: %.10f %.10f",
                need_print[name], layer.gradInput:std(), layer.gradInput:mean()
            ))
        end
    end

    local n = #net.modules

    for i = 1, n-2 do _print(net.modules[i]) end
    print('residual tower: ')
    _print_grad(net.modules[n-1])
    print('policy head: ')
    _print_grad(net.modules[n].modules[1])
    print('value head: ')
    _print_grad(net.modules[n].modules[2])
end

return util
