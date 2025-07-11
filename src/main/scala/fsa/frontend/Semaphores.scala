package fsa.frontend

import chisel3._
import chisel3.util._
import fsa.isa.ISA.Constants._

class Semaphore extends Bundle {
  val id = UInt(SEM_ID_BITS.W)
  val value = UInt(SEM_VALUE_BITS.W)
}


class Semaphores(nRead: Int, nWrite: Int) extends Module {
  val io = IO(new Bundle {
    val acquire = Vec(nRead, Flipped(Decoupled(new Semaphore)))
    val release = Vec(nWrite, Flipped(Valid(new Semaphore)))
  })

  val semaphores = RegInit(VecInit(Seq.fill(N_SEMAPHORES){0.U(SEM_VALUE_BITS.W)}))
  val busy = RegInit(VecInit(Seq.fill(N_SEMAPHORES){ false.B }))

  io.release.foreach{ release =>
    when(release.fire) {
      busy(release.bits.id) := false.B
      semaphores(release.bits.id) := release.bits.value
    }
  }

  io.acquire.foreach{ acq =>
    acq.ready := !busy(acq.bits.id) && acq.bits.value === semaphores(acq.bits.id)
    when(acq.fire) {
      busy(acq.bits.id) := true.B
    }
  }

}
