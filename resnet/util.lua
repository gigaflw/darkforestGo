-- @Author: gigaflw
-- @Date:   2017-11-29 16:25:36
-- @Last Modified by:   gigaflw
-- @Last Modified time: 2017-11-30 13:45:13

require 'nn'  -- necessary because checkpoints use nn modeul
local CBoard = require 'board.board'
local common = require 'common.common'

function play(model, board_history, player)
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
    ]]
    local input = torch.FloatTensor(1, 17, 19, 19):zero()

    for i = 1, 8 do
        input[1][2*i-1] = CBoard.get_stones(board_history[i], player)
        input[1][2*i] = CBoard.get_stones(board_history[i], CBoard.opponent(player))
    end

    input:narrow(2, 17, 1):fill(player == common.black and 1 or 0)
    output = model:forward(input)
    output[1] = nn.SoftMax():forward(output[1]:double())
    return output
end

function demo()
    local net = torch.load('./resnet.ckpt/latest.params')

    net = net.net
    board_history = {}
    for i = 1, 8 do
        board_history[i] = CBoard.new()
        CBoard.play(board_history[i], math.random(19), math.random(19), math.random(2) == 1 and common.black or common.white)
    end
    out = play(net, board_history, common.black)
end
