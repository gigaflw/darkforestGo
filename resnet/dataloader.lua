-- @Author: gigaflw
-- @Date:   2017-11-21 20:08:59
-- @Last Modified by:   gigaflw
-- @Last Modified time: 2017-12-20 11:11:52

local tnt = require 'torchnet'
local sgf = require 'utils.sgf'
local goutils = require 'utils.goutils'
local common = require 'common.common'
local CBoard = require 'board.board'
local argcheck = require 'argcheck'
local resnet_utils = require 'resnet.utils'

local om = require 'board.ownermap'
local dp_v2 = require 'board.pattern_v2'
local _owner_map = om.new()
local _def_policy = dp_v2.init('models/playout-model.bin', 'jp')
dp_v2.set_sample_params(_def_policy, -1, 0.125)

local parse_and_put = argcheck{
    doc = [[
        put the augmented (rotated) stone onto the board,
        get the input features and corresponding labels of a single play in a game,
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
    -- {name='last_features', type='torch.FloatTensor', help='Feature from last iteration since history info is needed'},
    {name='do_estimate', type='boolean', help='Whether use default policy to estimate the score if there is no clear one'},
    {name='augment', type='number', opt=true, help='[0, 7], the rotation style used for data augmentation, nil to disable'},
    call = function (board, game, do_estimate, augment)
        local x, y, player = sgf.parse_move(game.sgf[game.ply])
        local is_pass = x == 0 and y == 0

        if augment ~= nil then x, y = goutils.rotateMove(x, y, augment) end

        local moveIdx = is_pass and 19*19+1 or goutils.xy2moveIdx(x, y)
        assert(moveIdx > 0 and moveIdx <= 19 * 19 + 1)

        local winner = game:get_result_enum()

        local score = tonumber(game:get_result():sub(3))
        if score then
            score = score * (winner == player and 1 or -1) / 361
        elseif do_estimate then
            score, _, _, _ = om.util_compute_final_score(
                _owner_map, board, game:get_komi(), nil,
                function (b, max_depth) return dp_v2.run(_def_policy, b, max_depth, false) end
            )
            score = score / 361
        end

        local s = resnet_utils.board_to_features(board, player)
        local a = moveIdx
        local z = score or (winner == common.res_unknown and 0 or (winner == player and 1 or -1))

        if not is_pass then CBoard.play(board, x, y, player) end

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

        if opt.max_batches == -1 then
            opt.max_batches = math.floor(dataset:size() / opt.batch_size)
            assert(opt.max_batches > 0,  string.format(
                "Too small a dataset with size %d against batch size %d", dataset:size(), opt.batch_size)
            )
            print("opt.max_batches is adapted to "..opt.max_batches)
        end

        local game = nil
        local game_idx = 0
        local board = CBoard.new()

        local last_features = torch.FloatTensor(opt.n_feature, 19, 19) -- no use now
        local augment = nil -- augment style should be consistent during a single game

        if opt.debug then print("Dataloader in debug mode!") end

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
            local load = opt.debug and load_next_game or 
                ({sample = load_random_game, traverse = load_next_game})[opt.style]
            if game == nil or game.ply - 1 >= game:num_round() then
                repeat load() until game:num_round() > 0
            end
            game.ply = game.ply + 1

            -- this function should also put the augmented stone onto the board
            -- return parse_and_put(board, game, last_features, augment)
            return parse_and_put(board, game, opt.do_estimate, augment)
        end

        local data_pool = {}
        local pool_size = opt.debug and -1 or opt.data_pool_size
        local function _get_data_from_pool()
            if pool_size == -1 then return _parse_next_position()() end

            if #data_pool == 0 then
                for i = 1, pool_size do data_pool[i] = _parse_next_position() end
            end

            local choice = math.random(1, pool_size)
            local ret = data_pool[choice]()
            data_pool[choice] = _parse_next_position()
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
            if ind > max_batches then return nil end

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
