#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sstream>
#include <iostream>
#include "msaga_tsi.h"

void msaga_tsi_t::msaga_host_thread(void *arg) {
    msaga_tsi_t *tsi = static_cast<msaga_tsi_t*>(arg);
    tsi->run();
}

msaga_tsi_t::msaga_tsi_t(int argc, char** argv) : tsi_t(argc, argv),
    mem_dump_filename(""),
    mem_dump_start_addr(0),
    mem_dump_size(0)
{
    msaga_host.init(msaga_host_thread, this);
    std::vector<std::string> args(argv + 1, argv + argc);
    for (auto& arg : args) {
        if (arg.find("+dump-mem=") == 0) {
            std::string value = arg.substr(std::string("+dump-mem=").size());

            std::stringstream ss(value);
            std::string start_str, size_str;

            // Split the string using ':' as delimiter
            if (std::getline(ss, mem_dump_filename, ':') &&
                std::getline(ss, start_str, ':') &&
                std::getline(ss, size_str, ':')) {

                // Convert address and size from hex string to integers
                mem_dump_start_addr = std::stoul(start_str, nullptr, 16);
                mem_dump_size = std::stoul(size_str, nullptr, 16);

                // std::cout << "[MSAGA] Filename: " << mem_dump_filename << "\n";
                // std::cout << "[MSAGA] Start Address: 0x" << std::hex << mem_dump_start_addr << "\n";
                // std::cout << "[MSAGA] Size: 0x" << std::hex << mem_dump_size << "\n";
            } else {
                std::cerr << "[MSAGA] Error parsing +dump-mem argument: " << value << "\n";
            }
        }
    }
}

void msaga_tsi_t::dump_memory() {
    int fd = open(mem_dump_filename.c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
    if (fd == -1) {
        throw std::runtime_error("could not open " + mem_dump_filename);
    }
    for (uint32_t addr = mem_dump_start_addr; addr < mem_dump_start_addr + mem_dump_size; addr += sizeof(uint32_t)) {
        uint32_t data;
        read_chunk(addr, sizeof(uint32_t), &data);
        // std::cout << "[MSAGA] Dumping memory at address 0x" << std::hex << addr << ": 0x" << data << "\n";
        write(fd, &data, sizeof(uint32_t));
    }
    close(fd);
    std::cout << "[MSAGA] Memory dump saved to " << mem_dump_filename << "\n";
}

void msaga_tsi_t::run() {
    reset();
    load_instructions(target_args()[0]);
    while(!should_exit()) {
        int state = msaga_tsi_t::STATE_IDLE;
        while (state != msaga_tsi_t::STATE_DONE) {
            read_chunk(0x8008, sizeof(int), &state);
        }
        if (!mem_dump_filename.empty()) {
            dump_memory();
        }
        htif_exit(0);
    }
    stop();
    while (true) {
        switch_to_target();
    }
}

void msaga_tsi_t::load_instructions(const std::string& path) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd == -1) {
        throw std::runtime_error(
            "could not open " + path + " (did you misspell it?");
    }

    // Get the file size
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        throw std::runtime_error(
            "Error getting the file size " + path);
    }

    size_t filesize = sb.st_size;
    if (filesize % sizeof(uint32_t) != 0) {
        throw std::runtime_error(
            "MSAGA instruction file size must be a multiple of 4! size: " + filesize
        );
    }
    // Memory-map the file
    void *mapped = mmap(nullptr, filesize, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) {
        throw std::runtime_error(
            "Error reading file " + path);
    }
    close(fd);

    uint32_t *data = static_cast<uint32_t *>(mapped);
    for (size_t i = 0; i < filesize / sizeof(uint32_t); i++) {
        write_chunk(0x8000, sizeof(uint32_t), data);
        data++;
    }
    if (munmap(mapped, filesize) == -1) {
        throw std::runtime_error("Error unmapping file: " + path);
    }
}


void msaga_tsi_t::reset() {
    uint32_t x = 0xffffffff;
    write_chunk(0x8004, 4, &x);
}
