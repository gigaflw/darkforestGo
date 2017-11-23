-- @Author: gigaflw
-- @Date:   2017-11-21 20:08:59
-- @Last Modified by:   gigaflw
-- @Last Modified time: 2017-11-24 16:22:02

torch.setdefaulttensortype('torch.FloatTensor')

local tnt = require 'torchnet'
local sgf = require 'utils.sgf'
local goutils = require 'utils.goutils'
local common = require 'common.common'
local CBoard = require 'board.board'
local argcheck = require 'argcheck'

-----------------------
-- Feature & Label Extraction
-----------------------
local get_features = argcheck{
    help = [[
        Given a 19x19 board and the current player
        extracts 17 feature planes
        return a 17 x 19 x 19 tensor
    ]],
    {name='board', type='cdata', help='A `board.board` instance'},
    {name='player', type='number', help='One of common.black ( = 1 ) or common.white ( = 2 )'},
    call = function (board, player)
        local ret = torch.FloatTensor(17, 19, 19)
        -- TODO: ....
        return ret
    end
}


local get_input_n_label = argcheck{
    help = [[
        get the input features and corresponding labels of a single play in a game,
        which position is parsed depends on `game.ply` attribute
        return: {
            s: the 17 x 19 x 19 feature tensor, input of the network
            a: an integer in [1, 19*19+1], denoting human experts' move, the extra '+1' means pass.
              Should it be a pass, a = 19*19+1
            z: 1 if the current player wins, -1 otherwise
        }
        NOTICE: only look 1 step foreward, unlike darkforestGo
    ]],
    {name='board', type='cdata', help='A `board.board` instance'},
    {name='game', type='sgfloader', help='A `sgfloader` instance, get by calling `sgf.parse()'},
    {name='augment', type='boolean', default=false, help='Carry out rotation/reflection on data'},
    call = function (board, game, augment)
        local x, y, player = sgf.parse_move(game.sgf[game.ply])
        
        ---------------------
        -- Data Augmentation
        ---------------------
        local transformStyle = 0
        if augment then
            transformStyle = torch.random(0, 7)
            feature = goutils.rotateTransform(feature, transformStyle)
        end
        local x_rot, y_rot = goutils.rotateMove(x, y, transformStyle)

        ---------------------
        -- Prepare Move
        ---------------------
        local is_pass = x == 0 and y == 0
        local moveIdx = is_pass and 19*19+1 or goutils.xy2moveIdx(x_rot, y_rot)
        assert(moveIdx > 0 and moveIdx <= 19 * 19 + 1)

        return {
          s = get_features(board, player),
          a = moveIdx,
          z = game:get_result_enum() == player and 1 or -1, -- FIXME: what if a tie?
        }
    end
}
-----------------------
-- Feature & Label Ends
-----------------------

get_dataloader = argcheck{
    help = [[
        bridge the sgf dataset and the inputs and labels required by network training
        usage:
        > d = get_dataloader('test', opt.batch_size)
        > d.load_random_game()
        > for ind, input, label in d do
        >   model:forward(input)
        > end
        where input is shaped batch_size x feature_planes x 19 x 19,
        label = { a = <a batch_size-d vector>, z = <a batch_size-d vector> }
        refer to `get_input_n_label` for meanings of features, `a` and `z`
    ]],
    {name = 'partition', type='string', help='"test" or "train"'},
    {name = 'batch_size', type='number'},
    call = function (partition, batch_size)
        local name = 'kgs_' .. partition
        local dataset = tnt.IndexedDataset{fields = { name }, path = './dataset'}
        local batch_size = batch_size
        
        local game = nil
        local game_idx = 0
        local board = CBoard.new()

        local function load_game(idx)
            local content = dataset:get(idx)[name].table.content
            game_idx = idx
            game = sgf.parse(content:storage():string(), name)
            game.ply = 1
            CBoard.clear(board)
            goutils.apply_handicaps(board, game)

            print(idx..'-th game is loaded')
            return game
        end
        local function load_random_game() return load_game(math.random(dataset:size())) end
        local function load_next_game() return load_game(game_idx < dataset:size() and game_idx + 1 or 1) end

        local function _parse_next_position()
            if game.ply >= game:num_round() then load_next_game() end
            game.ply = game.ply + 1

            local x, y, player = sgf.parse_move(game.sgf[game.ply])
            assert(player ~= nil, "Encounted nil player in "..game_idx.."-th game, "..game.ply.."-th move")
            CBoard.play(board, x, y, player)
            -- CBoard.show(board, 'last_move')

            return get_input_n_label(board, game)
        end


        -- reusing tensors to save memory
        local s = torch.FloatTensor(batch_size, 17, 19, 19)
        local a = torch.IntTensor(batch_size)
        local z = torch.FloatTensor(batch_size)

        local function _iter_batch(max_batches, ind)
            ind = ind + 1
            if ind == max_batches then return nil end

            s:zero(); a:zero(); z:zero()

            for i = 1, batch_size do
            -- for i, feature, label in _iter, game, 0 do
                local data = _parse_next_position()
                s[i]:copy(data.s) -- should use copy or `=` ?
                a[i] = data.a
                z[i] = data.z
            end

            return ind, s, {a = a, z = z}
        end

        return {
            load_game = load_game,
            load_random_game = load_random_game,
            iter = function(max_batches) return _iter_batch, max_batches, 0 end
        }
    end
}

return get_dataloader
