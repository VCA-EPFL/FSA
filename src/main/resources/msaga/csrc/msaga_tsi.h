#ifndef __MSAGA_TSI_H
#define __MSAGA_TSI_H

#include <stdexcept>

#include <fesvr/tsi.h>
#include <fesvr/htif.h>
#include <fesvr/context.h>

class msaga_tsi_t : public tsi_t
{
 public:
  msaga_tsi_t(int argc, char** argv);
  virtual ~msaga_tsi_t() = default;

  // hide the original `switch_host` to switch to new host `msaga_host`
  void switch_to_host() { msaga_host.switch_to(); }
  // hide the original `run` from htif
  void run();
  void idle() override { switch_to_target(); }

 protected:
  void reset() override;

 private:
  context_t msaga_host;

  static void msaga_host_thread(void *tsi);

};
#endif
