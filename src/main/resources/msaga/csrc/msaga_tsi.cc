#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "msaga_tsi.h"

void msaga_tsi_t::msaga_host_thread(void *arg) {
    msaga_tsi_t *tsi = static_cast<msaga_tsi_t*>(arg);
    tsi->run();
}

msaga_tsi_t::msaga_tsi_t(int argc, char** argv) : tsi_t(argc, argv)
{
    msaga_host.init(msaga_host_thread, this);
    for(auto t : target_args()) {
        printf("targs: %s\n", t.c_str());
    }
    for(auto h : host_args()) {
        printf("hargs: %s\n", h.c_str());
    }
}

void msaga_tsi_t::run() {
    load_instructions(target_args()[0]);
    while (!in_valid()) {
        switch_to_target();
    }
    reset();
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
