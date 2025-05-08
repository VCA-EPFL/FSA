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
    // TODO: push real instructions here
    uint32_t inst = 0x12345678;
    write_chunk(0x8000, 4, &inst);
    reset();
    while (true) {
        switch_to_target();
    }
}

void msaga_tsi_t::reset() {
    uint32_t x = 0xffffffff;
    write_chunk(0x8004, 4, &x);
}
