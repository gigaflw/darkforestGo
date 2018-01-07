--
-- Created by HgS_1217_
-- Date: 2018/1/7
--

local utils = {}

function utils.to_string_ex(value)
    if type(value) == 'table' then
        return utils.table_to_string(value)
    elseif type(value) == 'string' then
        return "\'" .. value .. "\'"
    else
        return tostring(value)
    end
end

function utils.table_to_string(t)
    if t == nil then return "" end
    local retstr = "{"

    local i = 1
    for key, value in pairs(t) do
        local signal = ","
        if i == 1 then
            signal = ""
        end

        if key == i then
            retstr = retstr .. signal .. utils.to_string_ex(value)
        else
            if type(key) == 'number' or type(key) == 'string' then
                retstr = retstr .. signal .. '[' .. utils.to_string_ex(key) .. "]=" .. utils.to_string_ex(value)
            else
                if type(key) == 'userdata' then
                    retstr = retstr .. signal .. "*s" .. utils.table_to_string(getmetatable(key)) .. "*e" .. "=" .. utils.to_string_ex(value)
                else
                    retstr = retstr .. signal .. key .. "=" .. utils.to_string_ex(value)
                end
            end
        end

        i = i + 1
    end

    retstr = retstr .. "}"
    return retstr
end

function utils.string_to_table(str)
    if str == nil or type(str) ~= "string" then
        return
    end

    return loadstring("return " .. str)()
end

return utils
