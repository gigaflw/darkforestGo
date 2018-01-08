--
-- Created by HgS_1217_
-- Date: 2018/1/7
--

package.path = package.path .. ';../?.lua'

local socket = require("socket")
local json = require("cjson")

local host = "127.0.0.1"
local port = 5000
local server = assert(socket.bind(host, port, 1024))
server:settimeout(0)
local select_timeout = 10
local client_tab = {}

print("Server Start " .. host .. ":" .. port)

local test = 0

while true do
    local conn = server:accept()
    if conn then
        table.insert(client_tab, conn)
        print("Client " .. tostring(conn) .. " successfully connect!")
    end

    for conn_count, client in pairs(client_tab) do
        local recvt, sendt, status = socket.select({client}, nil, select_timeout)
        if #recvt > 0 then
            local receive, receive_status = client:receive()
            if receive_status ~= "closed" then
                if receive then
                    local byte_receive = json.decode(receive)
                    print("Receive Client " .. tostring(client) .. " : ", byte_receive)

                    local file = io.open("pipe2.txt", "wb")
                    file:write(byte_receive)
                    file:close()


                    -- TODO: Do something
                    os.execute("sleep 4")
                    test = test + 1
                    print(test)


                    local res = {test = "test"}
                    assert(client:send(json.encode(res) .. "\n"))
                end
            else
                table.remove(client_tab, conn_count)
                client:close()
                print("Client " .. tostring(client) .. " disconnect!")
            end
        end
    end
end
