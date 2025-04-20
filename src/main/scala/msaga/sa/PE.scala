package msaga.sa

import chisel3._
import chisel3.util._
import ArithmeticSyntax._

object MacCMD {
  def width = 2
  def ADD = 0.U(width.W)
  def SUB = 1.U(width.W)
  def MAC = 2.U(width.W)
  def EXP = 3.U(width.W)
}

abstract class MacUnit[E <: Data : Arithmetic, A <: Data : Arithmetic](elemType: E, accType: A) extends Module {
  val io = IO(new Bundle {
    val in_a = Input(elemType) // reg in PE
    val in_b = Input(elemType) // left input
    val in_c = Input(accType) // up/down input
    val in_cmd = Input(UInt(MacCMD.width.W))
    val out = Output(accType)
  })
}

class PECtrl(cols: Int) extends Bundle {
  val acc = Bool()
  val sub = Bool()
  val add_sub_ui = Bool()
  val mac = Bool()
  val exp = Bool()
  val load_reg_li = UInt(log2Up(cols + 1).W)
  val load_reg_ui = Bool()
  val flow_lr = Bool()
  val flow_ud = Bool()
  val flow_reg_d = Bool()
}

class PE[E <: Data : Arithmetic, A <: Data : Arithmetic, MAC <: MacUnit[E, A]]
(cols: Int, elemType: E, accType: A, macGen: () => MAC) extends Module
{
  val io = IO(new Bundle {
    val in_ctrl = Flipped(Valid(new PECtrl(cols)))
    val out_ctrl = Valid(new PECtrl(cols))
    val u_input = Flipped(Valid(accType))
    val u_output = Valid(accType)
    val d_input = Flipped(Valid(accType))
    val d_output = Valid(accType)
    val l_input = Flipped(Valid(elemType))
    val r_output = Valid(elemType)
  })

  val macUnit = Module(macGen())

  val reg = Reg(elemType)
  val ctrl = io.in_ctrl.bits

  when(io.in_ctrl.fire) {
    when(ctrl.load_reg_li === 1.U) {
      reg := io.l_input.bits
    }.elsewhen(ctrl.load_reg_ui) {
      reg := io.u_input.bits
    }.elsewhen(ctrl.sub || ctrl.exp) {
      reg := macUnit.io.out
    }
  }

  val expReg = RegNext(ctrl.exp && io.in_ctrl.fire, false.B)
  val exp1 = ctrl.exp && !expReg
  val exp2 = ctrl.exp && expReg

  /*
    Exp take 2 cycles:
    cycle 0: y = x * log2(e)
    cycle 1: z = 2^y
  */

  macUnit.io.in_a := reg
  macUnit.io.in_b := Mux(exp1, elemType.lg2_e, io.l_input.bits)
  macUnit.io.in_c := Mux(exp1, accType.zero,
    Mux(ctrl.add_sub_ui,
      Mux(io.u_input.valid, io.u_input.bits, accType.zero),
      io.d_input.bits
    )
  )
  macUnit.io.in_cmd := Mux(ctrl.mac || exp1, MacCMD.MAC,
    Mux(ctrl.acc, MacCMD.ADD, Mux(ctrl.sub, MacCMD.SUB, MacCMD.EXP))
  )

  io.out_ctrl := io.in_ctrl
  io.out_ctrl.bits.load_reg_li := ctrl.load_reg_li - 1.U

  io.r_output.bits := io.l_input.bits
  io.r_output.valid := io.in_ctrl.fire && ctrl.flow_lr

  io.u_output.bits := macUnit.io.out
  io.u_output.valid := io.in_ctrl.fire && ((ctrl.mac || ctrl.acc) && !ctrl.add_sub_ui)

  io.d_output.bits := Mux(ctrl.flow_ud, io.u_input.bits,
    Mux(ctrl.flow_reg_d, reg, macUnit.io.out)
  )
  io.d_output.valid := io.in_ctrl.fire && (ctrl.flow_ud || ctrl.flow_reg_d ||
    (ctrl.mac || ctrl.acc) && ctrl.add_sub_ui
  )

}
