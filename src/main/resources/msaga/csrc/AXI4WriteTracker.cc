#include <svdpi.h>
#include <vpi_user.h>
#include <stdio.h>
#include <stdint.h>
#include <algorithm>
#include <cassert>
#include <deque>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>

#include "msaga_tsi.h"

namespace msaga
{

    extern std::vector<MemDump> mem_dumps;

    struct AWInfo
    {
        uint64_t addr;
        uint8_t size;
        uint8_t len;
    };

    std::deque<AWInfo> aw_info;
    void sv_to_uint8_vector(
        const svOpenArrayHandle arr, std::vector<uint8_t> &dst, int offset)
    {
        int byte_count = svSize(arr, 1); // total number of bits
        for (int i = 0; i < byte_count; ++i)
        {
            svGetBitArrElemVecVal((svBitVecVal *)&dst[i + offset], arr, i);
        }
    }

    bool initialized = false;

    void axi_tracker_init()
    {
        s_vpi_vlog_info info;
        assert(vpi_get_vlog_info(&info) && "Failed to get VPI vlog info");
        for (int i = 1; i < info.argc; i++)
        {
            std::string arg(info.argv[i]);
            if (arg.find("+dump-mem=") == 0)
            {
                std::string value = arg.substr(std::string("+dump-mem=").size());
                std::stringstream ss(value);
                std::string start_str, size_str;
                MemDump mem_dump;
                // Split the string using ':' as delimiter
                if (std::getline(ss, mem_dump.filename, ':') &&
                    std::getline(ss, start_str, ':') &&
                    std::getline(ss, size_str, ':'))
                {
                    mem_dump.start_addr = std::stoul(start_str, nullptr, 16);
                    mem_dump.size = std::stoul(size_str, nullptr, 16);
                    mem_dump.data.resize(mem_dump.size);
                    printf("[MSAGA] Memory dump configured: %s, start=0x%08lx, size=%lu\n",
                           mem_dump.filename.c_str(), mem_dump.start_addr, mem_dump.size);
                    mem_dumps.emplace_back(mem_dump);
                }
                else
                {
                    std::cerr << "[MSAGA] Error parsing +dump-mem argument: " << value << "\n";
                }
            }
        }
        initialized = true;
    }

}

using namespace msaga;

extern "C" void axi_tracker_dpi(
    svBit aw_fire,
    uint64_t aw_addr,
    uint8_t aw_size,
    uint8_t aw_len,
    svBit w_fire,
    const svOpenArrayHandle w_data,
    svBit w_last)
{
    if (!initialized)
    {
        axi_tracker_init();
    }

    if (aw_fire)
    {
        // printf("AW: addr=0x%08x, size=%u, len=%u\n", aw_addr, aw_size, aw_len);
        aw_info.push_back({.addr = aw_addr,
                           .size = aw_size,
                           .len = aw_len});
    }

    if (w_fire)
    {
        // printf("W fire, awinfo size: %zu\n", aw_info.size());
        assert(!aw_info.empty() && "Write without preceding AW");
        // check address of the write
        uint64_t waddr = aw_info.front().addr;
        for (auto &dump : mem_dumps)
        {
            if (waddr >= dump.start_addr && waddr < dump.start_addr + dump.size)
            {
                size_t offset = waddr - dump.start_addr;
                // printf("[MSAGA] Writing to memory dump %s at offset %zu (addr=0x%08lx)\n",
                //       dump.filename.c_str(), offset, waddr);
                sv_to_uint8_vector(w_data, dump.data, offset);
                // printf("[MSAGA] Data written to memory dump %s\n", dump.filename.c_str());
            }
        }
        if (w_last)
        {
            aw_info.pop_front();
        }
        else
        {
            aw_info.front().addr += (1 << aw_info.front().size);
        }
    }
}
