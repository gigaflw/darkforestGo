-- @Author: gigaflw
-- @Date:   2017-11-21 20:08:59
-- @Last Modified by:   gigaflw
-- @Last Modified time: 2018-01-01 20:01:08

local tnt = require 'torchnet'
local sgf = require 'utils.sgf'
local goutils = require 'utils.goutils'
local common = require 'common.common'
local CBoard = require 'board.board'
local argcheck = require 'argcheck'
local resnet_utils = require 'resnet.utils'

local parse_and_put = argcheck{
    doc = [[
        get the input features and corresponding labels of a single play in a game,
        then put the augmented (rotated) stone onto the board
        which position is parsed depends on `game.ply` attribute
        return: {
            s: the 12 x 19 x 19 feature tensor, input of the network
            a: an integer in [1, 19*19+1], denoting human experts' move, the extra '+1' means pass.
              Should it be a pass, a = 19*19+1
            z:
                If we have a number on how many scores the current player won, z = score / 361
                else, if `do_estimate` is given, continue the game and estimate the score with default policy
                    this can be SLOW
                else, if we know who wins, z = 1 if the current player wins, -1 if the opponent
                else, if tie or time, z = 0
        }
        NOTICE: only look 1 step foreward, unlike darkforestGo
    ]],
    {name='board', type='cdata', help='A `board.board` instance'},
    {name='game', type='sgfloader', help='A `sgfloader` instance, get by calling `sgf.parse()'},
    {name='augment', type='number', opt=true, help='[0, 7], the rotation style used for data augmentation, nil to disable'},
    {name='no_pass', type='boolean'},
    call = function (board, game, augment, no_pass)
        local x, y, player = sgf.parse_move(game.sgf[game.ply])
        local is_pass = x == 0 and y == 0

        if is_pass and no_pass then return false end

        if augment ~= nil then x, y = goutils.rotateMove(x, y, augment) end

        local moveIdx = is_pass and 19*19+1 or goutils.xy2moveIdx(x, y)
        assert(moveIdx > 0 and moveIdx <= 19 * 19 + 1)

        local winner = game:get_result_enum()

        local s = resnet_utils.board_to_features(board, player)
        local a = moveIdx
        local z = winner == common.res_unknown and 0 or (winner == player and 1 or -1)

        CBoard.play(board, x, y, player)

        return function()
            return { s = s, a = a, z = z }
        end
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
        Dataset must be saved in torchnet.IndexedDataset format

        usage:
        > d = get_dataloader('./dataset/kgs_test', opt.batch_size)
        -- daatset/kgs_test.bin & daatset/kgs_test.idx must exist
        -- refer to https://github.com/torchnet/torchnet for detail

        > d.load_random_game()
        > for ind, input, label in d do
        >   model:forward(input)
        > end
        where input is shaped batch_size x feature_planes x 19 x 19,
        label = { a = <a batch_size-d vector>, z = <a batch_size-d vector> }
        refer to `parse_and_put` for meanings of features, `a` and `z`
    ]],
    {name = 'dataset_path', type='string', help='relative path to the torchnet.IndexedDataset file'},
    {name = 'opt', type='table'},
    call = function (dataset_path, opt)
        for _, name in pairs{'batch_size', 'data_augment', 'data_pool_size'} do
            assert(opt[name] ~= nil, "No '"..name.."' found in opt")
        end
        math.randomseed(os.time())

        local dataset_name, dataset_dir = paths.basename(dataset_path), paths.dirname(dataset_path)
        local dataset = tnt.IndexedDataset{fields = { dataset_name }, path = dataset_dir}

        local game = nil
        local game_idx = 0          -- index of the current game
        local game_cnt = 0          -- how many games have been loaded in total (include invalid ones)
        local board = CBoard.new()

        local augment = nil -- augment style should be consistent during a single game

        if opt.debug then print("Dataloader in debug mode!") end
        print("Dataloader in "..opt.style.." mode")
        if opt.max_batches == -1 then print("Dataloader will traverse all data") end

        -----------------------
        -- loading games
        -----------------------
        local function _get_sgf_string(idx)
            -- how we parse dataset is decided by how it is created
            -- for 'kgs_test' & 'kgs_train', they are created by facebook team
            -- for others, they are created in 'resnet.rl_train._save_sgf_to_dataset'
            -- here is a tight coupling, use dependency injection to solve it
            --  if more custom dataset are to be added
            if dataset_name == 'kgs_test' or dataset_name == 'kgs_train' then
                return dataset:get(idx)[dataset_name].table.content:storage():string()
            else
                return dataset:get(idx)[dataset_name].sgf
            end
        end

        local use_augment = not opt.debug and opt.data_augment
        local function load_game(idx)
            local sgf_string = _get_sgf_string(idx)

            game_idx = idx
            game = sgf.parse(sgf_string, dataset_name)
            game.ply = 1
            game_cnt = game_cnt + 1
            CBoard.clear(board)

            if use_augment then
                augment = torch.random(0, 7)  -- according to goutil.rotateMove and goutil.rotateTransform
            end

            goutils.apply_handicaps(board, game)
            return game
        end
        local function load_random_game() return load_game(math.random(dataset:size())) end
        local function load_next_game() return load_game(game_idx < dataset:size() and game_idx + 1 or 1) end

        -----------------------
        -- iterator interface
        -----------------------
        local function _parse_next_position()
            -- load new game if necessary
            if game == nil or game.ply > game:num_round() - 3 then  -- game.sgf[2] is the first, game.sgf[game:num_round()] the last
                 local load = opt.debug and load_next_game or 
                    ({sample = load_random_game, traverse = load_next_game})[opt.style]
                repeat load() until game:num_round() > opt.min_ply and not (opt.no_tie and game:get_result_enum() == common.res_unknown)
                if opt.verbose then
                    print(string.format('%d-th game is loaded, rounds: %d, augment: %s', game_idx, game:num_round(), augment))
                end
            end

            while game.ply < game:num_round() - 3 do
                local exceed_min_ply = game.ply > opt.min_ply
                local skip_this_one = opt.dropout > math.random()

                if exceed_min_ply and not skip_this_one then break end

                game.ply = game.ply + 1
                CBoard.play(board, x, y, player)
            end

            game.ply = game.ply + 1
            return parse_and_put(board, game, augment, opt.no_pass)
        end

        local data_pool = {}
        local pool_size = opt.debug and -1 or opt.data_pool_size
        local function _get_data_from_pool()
            function _get_next_valid_cb()
                local cb
                repeat cb = _parse_next_position() until cb ~= false
                return cb
            end

            if pool_size == -1 then return _get_next_valid_cb()() end

            if #data_pool == 0 then
                for i = 1, pool_size do data_pool[i] = _get_next_valid_cb() end
            end

            local choice = math.random(1, pool_size)
            local ret = data_pool[choice]()
            data_pool[choice] = _get_next_valid_cb()
            return ret
        end

        -- reusing tensors to save memory
        local s = torch.FloatTensor(opt.batch_size, opt.n_feature, 19, 19)
        local a = torch.FloatTensor(opt.batch_size)
        local z = torch.FloatTensor(opt.batch_size)
        local shuffle = {}
        for i = 1, opt.batch_size do shuffle[i] = i end

        local function _iter_batch(max_batches, ind)
            ind = ind + 1
            if (max_batches ~= -1 and ind > max_batches) or
                (max_batches == -1 and game_cnt > dataset:size()) then  -- -1 means traverse all data
                game_cnt = 0
                return nil
            end

            s:zero(); a:zero(); z:zero()

            if not opt.debug then
                -- in-batch shuffle
                for i = 1, opt.batch_size do
                    j = math.random(i, opt.batch_size)
                    shuffle[i], shuffle[j] = shuffle[j], shuffle[i]
                end
            end

            for i = 1, opt.batch_size do
                local data = _get_data_from_pool()
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
