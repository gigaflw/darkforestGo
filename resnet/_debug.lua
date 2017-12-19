util = require 'resnet.utils'
sgf = require 'utils.sgf'
goutil = require 'utils.goutils'
tnt = require 'torchnet'
B = require 'board.board'
common = require 'common.common'

obj = torch.load('resnet.ckpt/latest.cpu.params')
model = obj.net

dataset = tnt.IndexedDataset{fields = { 'kgs_test' }, path = './dataset'}
content = dataset:get(1)['kgs_test'].table.content
game = sgf.parse(content:storage():string(), '')

get_move = function(idx) return sgf.parse_move(game.sgf[idx]) end

b = B.new()

B.play(b, get_move(2))

model:evaluate()

input = util.board_to_feature(b, common.white)
output = model:forward(input:resize(1, 12, 19, 19))