#ifndef __fsa_TSI_H
#define __fsa_TSI_H

#include <stdexcept>

#include <fesvr/tsi.h>
#include <fesvr/htif.h>
#include <fesvr/context.h>


class fsa_tsi_t : public tsi_t
{
 public:
  static const int STATE_IDLE = 0;
  static const int STATE_ACTIVE = 1;
  static const int STATE_DONE = 2;
  fsa_tsi_t(int argc, char** argv);
  virtual ~fsa_tsi_t() noexcept override = default;

  // hide the original `switch_host` to switch to new host `fsa_host`
  void switch_to_host() { fsa_host.switch_to(); }
  // hide the original `run` from htif
  void run();
  void idle() override { switch_to_target(); }

 protected:
  void reset() override;
  void load_instructions(const std::string& path);

 private:
  context_t fsa_host;

  static void fsa_host_thread(void *tsi);

};
#endif
