--
-- Created by HgS_1217_
-- Date: 2017/11/27
--

local utils = require("utils.utils")
local board = require("board.board")
local om = require("board.ownermap")
local dp = require("pachi_tactics.moggy")
local dcnn_utils = require("board.dcnn_utils")
local sgf = require("utils.sgf")
local common = require("common.common")

local self_play_mcts = {}



return self_play_mcts