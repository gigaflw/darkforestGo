-- @Author: gigaflw
-- @Date:   2017-12-17 10:06:29
-- @Last Modified by:   gigaflw
-- @Last Modified time: 2017-12-17 10:17:17

local doc = [[
    This patch code aims to stress the conflict between argcheck 2.0 and torch.class.
    In argcheck 2.0, if torch is installed, there would be a regex matching feature for type names like this:
    f = argcheck{ {name = 'x', type = 'torch.*Tensor'}, ... }
    which would be totally useless as soon as one `require 'class'`,
    which will rashly replace `argcheck.env.istype` to be `class.istype`,
    this will cause error if this feature is used.

    run following code before any `require 'class'` takes place will solve the problem
    usage:
        require '_torch_class_patch'
    or
        dofile '_torch_class_patch.lua'
]]

local env = require 'argcheck.env'
env._istype = env.istype

local class = require 'class'
class._istype = class.istype

class.istype = function (...) return env._istype(...) or class._istype(...) end
env.istype = class.istype
