package msaga.tsi

import chisel3._
import chisel3.util._
import chisel3.experimental.IntParam
import testchipip.tsi.TSIIO

class MSAGASimTSI(chipId: Int) extends BlackBox(Map("CHIPID" -> IntParam(chipId))) with HasBlackBoxResource {
  val io = IO(new Bundle {
    val clock = Input(Clock())
    val reset = Input(Bool())
    val tsi = Flipped(new TSIIO)
    val exit = Output(UInt(32.W))
  })
  addResource("/msaga/vsrc/MSAGASimTSI.v")
  addResource("/msaga/csrc/MSAGASimTSI.cc")
  addResource("/msaga/csrc/msaga_tsi.cc")
  addResource("/msaga/csrc/msaga_tsi.h")
}

object MSAGASimTSI {
  def connect(tsi: Option[TSIIO], clock: Clock, reset: Reset, chipId: Int = 0): Bool = {
    val exit = tsi.map { s =>
      val sim = Module(new MSAGASimTSI(chipId))
      sim.io.clock := clock
      sim.io.reset := reset
      sim.io.tsi <> s
      sim.io.exit
    }.getOrElse(0.U)
    val success = exit === 1.U
    val error = exit >= 2.U
    assert(!error, "*** FAILED *** (exit code = %d)\n", exit >> 1.U)
    success
  }
}
