--
-- Created by HgS_1217_
-- Date: 2018/1/7
--

package.path = package.path .. ';../?.lua'

local socket = require("socket")
local utils = require("net_socket.utils")

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
                    local table_receive = utils.string_to_table(receive)
                    print("Receive Client " .. tostring(client) .. " : ", table_receive)


                    -- TODO: Do something
                    os.execute("sleep 4")
                    test = test + 1
                    print(test)


                    local res = {test = "test"}
                    assert(client:send(utils.table_to_string(res) .. "\n"))
                end
            else
                table.remove(client_tab, conn_count)
                client:close()
                print("Client " .. tostring(client) .. " disconnect!")
            end
        end
    end
end
