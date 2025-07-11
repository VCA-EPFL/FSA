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
#include <mutex>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#include "fsa_tsi.h"

namespace fsa
{

    struct MemDump
    {
        uint64_t start_addr;
        uint64_t size;
        // not necessary to keep filename, but useful for debugging
        std::string filename;
        // data maybe accessed by multiple threads
        std::mutex mutex;
        uint8_t *data;
        int fd;

        // -> mutex makes MemDump non-copyable and non-movable, no need to worry about copy/move
        MemDump(const std::string &cmd)
        {
            std::stringstream ss(cmd);
            std::string start_str, size_str;
            // Split the string using ':' as delimiter
            if (std::getline(ss, filename, ':') &&
                std::getline(ss, start_str, ':') &&
                std::getline(ss, size_str, ':'))
            {
                start_addr = std::stoul(start_str, nullptr, 16);
                size = std::stoul(size_str, nullptr, 16);
                fd = open(filename.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0666);
                if (fd < 0)
                {
                    std::cerr << "[fsa] Error opening file: " << filename << "\n";
                    abort();
                }
                if (ftruncate(fd, size) < 0)
                {
                    std::cerr << "[fsa] Error truncating file: " << filename << "\n";
                    close(fd);
                    abort();
                }
                data = (uint8_t *)mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
                if (data == MAP_FAILED)
                {
                    std::cerr << "[fsa] Error memory-mapping file: " << filename << "\n";
                    close(fd);
                    abort();
                }
                printf("[fsa] Memory dump configured: %s, start=0x%08lx, size=%lu\n",
                       filename.c_str(), start_addr, size);
            }
            else
            {
                std::cerr << "[fsa] Error parsing +dump-mem argument: " << cmd << "\n";
                abort();
            }
        };

        ~MemDump()
        {
            if (data && data != MAP_FAILED)
            {
                munmap(data, size);
            }
            if (fd >= 0)
            {
                close(fd);
            }
        }
    };

    struct AWInfo
    {
        uint64_t addr;
        uint8_t size;
        uint8_t len;
    };

    // mem_dumps is shared among all trackers
    std::vector<std::shared_ptr<MemDump>> mem_dumps;
    // only use the first tracker to initialize mem_dumps
    std::mutex init_mutex;
    bool mem_dump_created = false;
}

using namespace fsa;

// return the pointer to AWInfo of the current tracker
extern "C" void* axi_tracker_init()
{
    auto aw_info = new std::deque<AWInfo>();

    {
        std::lock_guard<std::mutex> lock(init_mutex);
        if (mem_dump_created)
        {
            return (void *)aw_info; // already initialized
        }
        mem_dump_created = true;
    }

    s_vpi_vlog_info info;
    assert(vpi_get_vlog_info(&info) && "Failed to get VPI vlog info");
    for (int i = 1; i < info.argc; i++)
    {
        std::string arg(info.argv[i]);
        if (arg.find("+dump-mem=") == 0)
        {
            std::string value = arg.substr(std::string("+dump-mem=").size());
            auto mem_dump = std::make_shared<MemDump>(value);
            mem_dumps.emplace_back(mem_dump);
        }
    }
    return (void *)aw_info;
}

extern "C" void axi_tracker_dpi(
    void* aw_info_ptr,
    svBit aw_fire,
    uint64_t aw_addr,
    uint8_t aw_size,
    uint8_t aw_len,
    svBit w_fire,
    const svOpenArrayHandle w_data,
    svBit w_last)
{
    auto aw_info = (std::deque<AWInfo> *)aw_info_ptr;

    if (aw_fire)
    {
        aw_info->emplace_back(AWInfo{aw_addr, aw_size, aw_len});
    }

    if (w_fire)
    {
        assert(!aw_info->empty() && "Write without preceding AW");
        // check address of the write
        uint64_t waddr = aw_info->front().addr;
        for (auto &dump : mem_dumps)
        {
            if (waddr >= dump->start_addr && waddr < dump->start_addr + dump->size)
            {
                size_t offset = waddr - dump->start_addr;
                int bytes = svSize(w_data, 1);
                /* create a local buffer to hold the wdata here,
                   seems that svGetBitArrElemVecVal can not write to
                   a shared memory region directly.
                */
                std::vector<uint8_t> w_data_arr(bytes);
                for (int i = 0; i < bytes; ++i)
                {
                    svGetBitArrElemVecVal((svBitVecVal *)(w_data_arr.data() + i), w_data, i);
                }
                // transfer local buffer to shared memory
                std::lock_guard<std::mutex> lock(dump->mutex);
                memcpy(dump->data + offset, w_data_arr.data(), bytes);
            }
        }
        if (w_last)
        {
            assert(aw_info->front().len == 0 && "Write with WLAST but AW length not zero");
            aw_info->pop_front();
        }
        else
        {
            aw_info->front().addr += (1 << aw_info->front().size);
            aw_info->front().len--;
        }
    }
}
