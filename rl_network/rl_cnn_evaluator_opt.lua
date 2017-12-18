--
-- Created by HgS_1217_
-- Date: 2017/11/29
--

local pl = require 'pl.import_into'()

local opt = pl.lapp[[
  -g,--gpu  (default 2)                      GPU id to use.
  --async                                    Make it asynchronized.
  --pipe_path (default "../../dflog")        Path for pipe file. Default is in the current directory, i.e., go/mcts
  --codename  (default "darkfores2")         Code name for the model to load.
  --use_local_model                          If true, load the local model.
]]

return opt
