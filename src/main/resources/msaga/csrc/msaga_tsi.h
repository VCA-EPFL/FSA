#ifndef __MSAGA_TSI_H
#define __MSAGA_TSI_H

#include <stdexcept>

#include <fesvr/tsi.h>
#include <fesvr/htif.h>
#include <fesvr/context.h>

class msaga_tsi_t : public tsi_t
{
 public:
  static const int STATE_IDLE = 0;
  static const int STATE_ACTIVE = 1;
  static const int STATE_DONE = 2;
  msaga_tsi_t(int argc, char** argv);
  virtual ~msaga_tsi_t() = default;

  // hide the original `switch_host` to switch to new host `msaga_host`
  void switch_to_host() { msaga_host.switch_to(); }
  // hide the original `run` from htif
  void run();
  void idle() override { switch_to_target(); }

 protected:
  void reset() override;
  void load_instructions(const std::string& path);

 private:
  context_t msaga_host;
  std::string mem_dump_filename;
  uint32_t mem_dump_start_addr;
  uint32_t mem_dump_size;

  static void msaga_host_thread(void *tsi);
  void dump_memory();

};
#endif
