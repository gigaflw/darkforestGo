--
-- Created by HgS_1217_
-- Date: 2018/1/7
--

local socket = require("socket")
local utils = require("net_socket.utils")

local host = "127.0.0.1"
local port = 5000
local sock = assert(socket.connect(host, port))
sock:settimeout(0)
local select_timeout = 10

local test = 0

local input, recvt, sendt, status
while true do
    input = {q = 123, hgs = 1217 }
    if next(input) then
        assert(sock:send(utils.table_to_string(input) .. "\n"))
    end

    recvt, sendt, status = socket.select({sock}, nil, select_timeout)
    while #recvt > 0 do
        local response, receive_status = sock:receive()
        if receive_status ~= "closed" then
            if response then
                print(utils.string_to_table(response))


                -- TODO: Do something
                os.execute("sleep 2")
                test = test + 1
                print(test)


                recvt, sendt, status = socket.select({sock}, nil, 1)
            end
        else
            break
        end
    end
end
