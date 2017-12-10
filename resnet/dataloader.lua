-- @Author: gigaflw
-- @Date:   2017-11-21 20:08:59
-- @Last Modified by:   gigaflw
-- @Last Modified time: 2017-12-10 16:40:29

local tnt = require 'torchnet'
local sgf = require 'utils.sgf'
local goutils = require 'utils.goutils'
local common = require 'common.common'
local CBoard = require 'board.board'
local argcheck = require 'argcheck'
local resnet_util = require 'resnet.util'

local put_and_parse = argcheck{
    doc = [[
        put the augmented (rotated) stone onto the board,
        get the input features and corresponding labels of a single play in a game,
        which position is parsed depends on `game.ply` attribute
        return: {
            s: the 12 x 19 x 19 feature tensor, input of the network
            a: an integer in [1, 19*19+1], denoting human experts' move, the extra '+1' means pass.
              Should it be a pass, a = 19*19+1
            z: 1 if the current player wins, -1 otherwise
        }
        NOTICE: only look 1 step foreward, unlike darkforestGo
    ]],
    {name='board', type='cdata', help='A `board.board` instance'},
    {name='game', type='sgfloader', help='A `sgfloader` instance, get by calling `sgf.parse()'},
    -- {name='last_features', type='torch.FloatTensor', help='Feature from last iteration since history info is needed'},
    {name='augment', type='number', opt=true, help='[0, 7], the rotation style used for data augmentation, nil to disable'},
    call = function (board, game, augment)
        local x, y, player = sgf.parse_move(game.sgf[game.ply])
        local is_pass = x == 0 and y == 0

        if augment ~= nil then x, y = goutils.rotateMove(x, y, augment) end

        if not is_pass then CBoard.play(board, x, y, player) end
        local moveIdx = is_pass and 19*19+1 or goutils.xy2moveIdx(x, y)
        assert(moveIdx > 0 and moveIdx <= 19 * 19 + 1)

        local winner = game:get_result_enum()
        return {
          s = resnet_util.board_to_features(board, player),
          a = moveIdx,
          z = winner == common.res_unknown and 0 or (winner == player and 1 or -1)
        }
    end
}
-----------------------
-- Feature & Label Ends
-----------------------

-----------------------
-- Dataloader
-----------------------
get_dataloader = argcheck{
    doc = [[
        bridge the sgf dataset and the inputs and labels required by network training
        usage:
        > d = get_dataloader('test', opt.batch_size)
        > d.load_random_game()
        > for ind, input, label in d do
        >   model:forward(input)
        > end
        where input is shaped batch_size x feature_planes x 19 x 19,
        label = { a = <a batch_size-d vector>, z = <a batch_size-d vector> }
        refer to `put_and_parse` for meanings of features, `a` and `z`
    ]],
    {name = 'partition', type='string', help='"test" or "train"'},
    {name = 'opt', type='table'},
    call = function (partition, opt)
        batch_size = opt.batch_size
        use_augment = opt.data_augment
        assert(batch_size ~= nil, "No 'batch_size' found in opt")
        assert(use_augment ~= nil, "No 'data_augment' found in opt")

        math.randomseed(os.time())

        local name = 'kgs_' .. partition
        local dataset = tnt.IndexedDataset{fields = { name }, path = './dataset'}
        local batch_size = batch_size

        local game = nil
        local game_idx = 0
        local board = CBoard.new()

        -- reusing tensors to save memory
        local s = torch.FloatTensor(batch_size, opt.n_feature, 19, 19)
        local a = torch.FloatTensor(batch_size)
        local z = torch.FloatTensor(batch_size)
        local last_features = torch.FloatTensor(opt.n_feature, 19, 19)
        local augment = nil -- augment style should be consistent during a single game

        -----------------------
        -- loading games
        -----------------------
        local function load_game(idx)
            local content = dataset:get(idx)[name].table.content
            game_idx = idx
            game = sgf.parse(content:storage():string(), name)
            game.ply = 1
            CBoard.clear(board)

            last_features:zero()
            if use_augment then
                augment = torch.random(0, 7)  -- according to goutil.rotateMove and goutil.rotateTransform
            end

            goutils.apply_handicaps(board, game)

            if opt.verbose then
                print(string.format('%d-th game is loaded, rounds: %d, augment: %s', idx, game:num_round(), augment))
            end
            return game
        end
        local function load_random_game() return load_game(math.random(dataset:size())) end
        local function load_next_game() return load_game(game_idx < dataset:size() and game_idx + 1 or 1) end

        -----------------------
        -- iterator interface
        -----------------------
        local function _parse_next_position()
            if game == nil or game.ply - 1 >= game:num_round() then
                repeat load_random_game() until game:num_round() > 0
            end
            game.ply = game.ply + 1

            -- this function should also put the augmented stone onto the board
            -- return put_and_parse(board, game, last_features, augment)
            return put_and_parse(board, game, augment)
        end

        local function _iter_batch(max_batches, ind)
            ind = ind + 1
            if ind > max_batches then return nil end

            s:zero(); a:zero(); z:zero()
            local shuffle = {}
            
            for i = 1, batch_size do shuffle[i] = i end
            for i = 1, batch_size do
                j = math.random(i, batch_size)
                shuffle[i], shuffle[j] = shuffle[j], shuffle[i]
            end

            for i = 1, batch_size do
                local data = _parse_next_position()
                s[shuffle[i]] = data.s
                a[shuffle[i]] = data.a
                z[shuffle[i]] = data.z
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
