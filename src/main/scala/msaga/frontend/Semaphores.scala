package msaga.frontend

import chisel3._
import chisel3.util._

import msaga.isa.ISA.Constants._

class SemaphoreRead extends Bundle {
  val semaphoreId = Input(UInt(SEM_ID_BITS.W))
  val semaphoreValue = Input(UInt(SEM_VALUE_BITS.W))
  val ready = Output(Bool())
}

class SemaphoreWrite extends Bundle {
  val semaphoreId = UInt(SEM_ID_BITS.W)
  val semaphoreValue = UInt(SEM_VALUE_BITS.W)
}

class Semaphores(nRead: Int, nWrite: Int) extends Module {
  val io = IO(new Bundle {
    val read = Vec(nRead, new SemaphoreRead)
    val write = Vec(nWrite, Flipped(Valid(new SemaphoreWrite)))
  })

  val semaphores = RegInit(VecInit(Seq.fill(N_SEMAPHORES){0.U(SEM_VALUE_BITS.W)}))

  io.read.foreach{ r =>
    r.ready := r.semaphoreId === 0.U || semaphores(r.semaphoreId) === r.semaphoreValue
  }
  io.write.foreach{ w =>
    when(w.valid) {
      semaphores(w.bits.semaphoreId) := w.bits.semaphoreValue
    }
  }

}
