package msaga

import chisel3._
import chisel3.util._
import msaga.arithmetic.ArithmeticSyntax._
import msaga.arithmetic._

object AccumulatorCmd {
  def width = 2

  def EXP_S1 = 0.U(width.W)

  def EXP_S2 = 1.U(width.W)

  def ACC = 2.U(width.W)
}


class AccumulatorControl extends Bundle {
  val cmd = UInt(AccumulatorCmd.width.W)
}

class Accumulator[A <: Data : Arithmetic](cols: Int, accType: A, macGen: () => MacUnit[A, A]) extends Module {

  val io = IO(new Bundle {
    val ctrl_in = Flipped(Valid(new AccumulatorControl))
    val sa_in = Input(Vec(cols, accType))
    val sram_in = Input(Vec(cols, accType))
    val sram_out = Output(Vec(cols, accType))
  })

  val macUnit = Seq.fill(cols) {
    Module(macGen())
  }
  val scale = Seq.fill(cols) {
    Reg(accType)
  }
  val valid = io.ctrl_in.valid
  val cmd = io.ctrl_in.bits.cmd
  val exp_s1 = cmd === AccumulatorCmd.EXP_S1
  val exp_s2 = cmd === AccumulatorCmd.EXP_S2
  val acc = cmd === AccumulatorCmd.ACC

  /*
    * exp s1: scale <- sa_in * lg2e + 0
    * exp s2: scale <- pow2(scale)
    * accum: out <- scale * sram_in + sa_in
  */

  for (((((s, mac), sa_in), sram_in), sram_out) <- scale.zip(macUnit).zip(io.sa_in).zip(io.sram_in).zip(io.sram_out)) {
    mac.io.in_a := Mux(exp_s1, sa_in, s)
    mac.io.in_b := Mux(exp_s1, accType.lg2_e, sram_in)
    mac.io.in_c := Mux(exp_s1, accType.zero, sa_in)
    mac.io.in_cmd := Mux(exp_s2, MacCMD.EXP2, MacCMD.MAC)
    when(valid && (exp_s1 || exp_s2)) {
      s := mac.io.out
    }
    sram_out := mac.io.out
  }

}
