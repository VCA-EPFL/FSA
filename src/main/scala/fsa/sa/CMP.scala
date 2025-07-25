package fsa.sa

import chisel3._
import chisel3.util._
import fsa.arithmetic._
import fsa.arithmetic.ArithmeticSyntax._

object CmpControlCmd {
  def width = 3
  def UPDATE = 0.U(width.W)
  def PROP_MAX = 1.U(width.W)
  def PROP_MAX_DIFF = 2.U(width.W)
  def PROP_ZERO = 3.U(width.W)
  def RESET = 4.U(width.W)
  def PROP_EXP2_INTERCEPTS = 5.U(width.W)
}

class CmpControl extends Bundle {
  val cmd = UInt(CmpControlCmd.width.W)
}

class CMP[E <: Data : Arithmetic, A <: Data : Arithmetic](ev: ArithmeticImpl[E, A]) extends Module {
  val (accType, cmpUnitGen) = (ev.accType, ev.accCmp _)
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
  val do_reset = cmd === CmpControlCmd.RESET
  val prop_exp2_intercepts = cmd === CmpControlCmd.PROP_EXP2_INTERCEPTS
  val zero = accType.zero

  cmpUnit.io.in_a := Mux(update_new_max, io.d_input.bits, Mux(prop_new_max, zero, oldMax))
  cmpUnit.io.in_b := newMax

  val exp2_intercepts = VecInit(ev.exp2PwlIntercepts)
  val exp2_counter = Counter(exp2_intercepts.length)
  when(io.in_ctrl.fire && prop_exp2_intercepts) {
    exp2_counter.inc()
  }

  when(io.in_ctrl.fire) {
    when(do_reset) {
      newMax := accType.minimum
      oldMax := accType.minimum
    }.elsewhen(prop_zero || prop_exp2_intercepts) {
      // do nothing, keep newMax and oldMax unchanged
    }.otherwise({
      newMax := cmpUnit.io.out_max
      when(prop_diff) {
        oldMax := cmpUnit.io.out_max
      }
    })
  }

  val downCastDIn = ev.viewEasA(ev.cvtAtoE(io.d_input.bits))
  io.d_output.bits := Mux(prop_zero,
    zero,
    Mux(prop_exp2_intercepts,
      exp2_intercepts(exp2_counter.value),
      Mux(update_new_max,
        downCastDIn,
        cmpUnit.io.out_diff
      )
    )
  )
  io.d_output.valid := io.in_ctrl.valid && !do_reset
  io.out_ctrl := io.in_ctrl
}
