--
-- Created by HgS_1217_
-- Date: 2018/1/7
--

local socket = require("socket")
local json = require("cjson")

local host = "127.0.0.1"
local port = 5000
local sock = assert(socket.connect(host, port))
sock:settimeout(0)
local select_timeout = 10

local test = 0

local input, recvt, sendt, status
while true do
    input = {q = 123, hgs = 1217}

    local file = io.open("pipe.txt", "rb")
    input = file:read()
    file:close()

    if #input > 0 then
        assert(sock:send(json.encode(input) .. "\n"))
    end

    recvt, sendt, status = socket.select({sock}, nil, select_timeout)
    while #recvt > 0 do
        local response, receive_status = sock:receive()
        if receive_status ~= "closed" then
            if response then
                local json_receive = json.decode(response)
                print(json_receive)


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
