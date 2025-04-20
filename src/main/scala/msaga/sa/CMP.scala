package msaga.sa

import chisel3._
import chisel3.util._
import ArithmeticSyntax._

object CmpCMD {
  def width = 1
  def MAX = 0.U(width.W)
  def SUB = 1.U(width.W)
}

abstract class CmpUnit[A <: Data](accType: A) extends Module {
  val io = IO(new Bundle {
    val in_a = Input(accType)
    val in_b = Input(accType)
    val in_cmd = Input(UInt(CmpCMD.width.W))
    val out = Output(accType)
  })
}

class CmpControl extends Bundle {
  val update_new_max = Bool()
  val prop_new_max = Bool()
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

  val update_new_max = io.in_ctrl.bits.update_new_max
  val prop_new_max = io.in_ctrl.bits.prop_new_max
  val prop_delta_max = !update_new_max && !prop_new_max

  cmpUnit.io.in_a := newMax
  cmpUnit.io.in_b := Mux(update_new_max, io.d_input.bits, oldMax)
  cmpUnit.io.in_cmd := Mux(update_new_max, CmpCMD.MAX, CmpCMD.SUB)

  when(io.in_ctrl.fire) {
    when(update_new_max) {
      newMax := cmpUnit.io.out
    }.elsewhen(prop_delta_max) {
      oldMax := newMax
      newMax := accType.minimum
    }
  }

  io.d_output.bits := Mux(update_new_max, io.d_input.bits, Mux(prop_new_max, newMax, cmpUnit.io.out))
  io.d_output.valid := io.in_ctrl.valid
  io.out_ctrl := io.in_ctrl
}
