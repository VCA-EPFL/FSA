package msaga.utils

import chisel3._
import chisel3.util._

// Bluespec-style Ephemeral History Registers
class Ehr[T <: Data](n: Int, gen: T, init: Option[T]) extends Module {

  val io = IO(new Bundle {
    val read = Output(Vec(n, gen))
    val write = Flipped(Vec(n, Valid(gen)))
  })

  val reg = init.map(i => RegInit(i)).getOrElse(Reg(gen))

  io.read.head := reg
  io.read.tail.zip(io.write.init).foldLeft(reg){ case (r_last, (r, w)) =>
    r := Mux(w.valid, w.bits, r_last)
    r
  }

  val w = io.write.reduceRight((l, r) => Mux(r.valid, r, l))
  reg := Mux(w.valid, w.bits, reg)

  def read(idx: Int): T = io.read(idx)
  def write(idx: Int, v: T): Unit = {
    io.write(idx).valid := true.B
    io.write(idx).bits := v
  }

}

object Ehr {
  def apply[T <: Data](n: Int, gen: T, init: Option[T] = None) = {
    val ehr = Module(new Ehr(n, gen, init))
    ehr.io.write.foreach{ w =>
      w.valid := false.B
      w.bits := DontCare
    }
    ehr
  }
}