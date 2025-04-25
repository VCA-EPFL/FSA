package msaga.sa

import chisel3._
import chisel3.util._
import msaga.arithmetic._
import msaga.arithmetic.ArithmeticSyntax._

object CmpControlCmd {
  def width = 2
  def UPDATE = 0.U(width.W)
  def PROP_MAX = 1.U(width.W)
  def PROP_MAX_DIFF = 2.U(width.W)
  def PROP_ZERO = 3.U(width.W)
}

class CmpControl extends Bundle {
  val cmd = UInt(CmpControlCmd.width.W)
}

class CMP[A <: Data : Arithmetic](accType: A, cmpUnitGen: () => CmpUnit[A]) extends Module {
  val io = IO(new Bundle {
    val d_input = Flipped(Valid(accType))
    val d_output = Valid(accType)
    val in_ctrl = Flipped(Valid(new CmpControl))
    val out_ctrl = Valid(new CmpControl)
  })

  val cmpUnit = Module(cmpUnitGen())

  val oldMax = RegInit(accType.minimum)
  val newMax = RegInit(accType.minimum)

  val cmd = io.in_ctrl.bits.cmd
  val update_new_max = cmd === CmpControlCmd.UPDATE
  val prop_new_max = cmd === CmpControlCmd.PROP_MAX
  val prop_diff = cmd === CmpControlCmd.PROP_MAX_DIFF
  val prop_zero = cmd === CmpControlCmd.PROP_ZERO
  val zero = accType.zero

  cmpUnit.io.in_a := Mux(update_new_max, io.d_input.bits, Mux(prop_new_max, zero, oldMax))
  cmpUnit.io.in_b := newMax
  cmpUnit.io.in_cmd := Mux(update_new_max, CmpCMD.MAX, CmpCMD.SUB)

  when(io.in_ctrl.fire) {
    when(update_new_max) {
      newMax := cmpUnit.io.out
    }.elsewhen(prop_diff) {
      oldMax := newMax
      newMax := accType.minimum
    }
  }

  io.d_output.bits := Mux(prop_zero, zero, Mux(update_new_max, io.d_input.bits, cmpUnit.io.out))
  io.d_output.valid := io.in_ctrl.valid
  io.out_ctrl := io.in_ctrl
}
