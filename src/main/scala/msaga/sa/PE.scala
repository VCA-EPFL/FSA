package msaga.sa

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy._
import msaga.arithmetic._


class PECtrl extends Bundle {
  val mac = Bool()
  val acc_ui = Bool()
  val load_reg_li = Bool()
  val load_reg_ui = Bool()
  // pass through
  val flow_lr = Bool()
  val flow_ud = Bool()
  val flow_du = Bool()
  // mac_out -> reg
  val update_reg = Bool()
  // compute 2^reg
  val exp2 = Bool()

  // getElements might be dangerous, define them manually
  def getCtrlElements: Seq[Bool]= Seq(
    mac, acc_ui, load_reg_li, load_reg_ui,
    flow_lr, flow_ud, flow_du,
    update_reg, exp2
  )
}

@instantiable
class PE[E <: Data : Arithmetic, A <: Data : Arithmetic]
(ev: ArithmeticImpl[E, A]) extends Module
{
  val (accType, elemType, macGen) = (ev.accType, ev.elemType, ev.peMac _)
  @public val io = IO(new Bundle {
    val in_ctrl = Flipped(Valid(new PECtrl))
    val out_ctrl = Valid(new PECtrl)
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
  val fire = io.in_ctrl.fire

  val exp2Done = RegInit(false.B)
  when(fire) {
    when(ctrl.exp2) {
      exp2Done := exp2Done || macUnit.io.out_exp2
    }.otherwise({
      exp2Done := false.B
    })
  }

  when(fire) {
    when(ctrl.load_reg_li) {
      reg := io.l_input.bits
    }.elsewhen(ctrl.load_reg_ui) {
      reg := ev.viewAasE(io.u_input.bits)
    }.elsewhen(ctrl.update_reg || (macUnit.io.out_exp2 && !exp2Done)) {
      reg := macUnit.io.out_elemType
    }
  }

  macUnit.io.in_a := reg
  macUnit.io.in_b := io.l_input.bits
  macUnit.io.in_c := Mux(ctrl.acc_ui, io.u_input.bits, io.d_input.bits)
  macUnit.io.in_cmd := Mux(ctrl.exp2, MacCMD.EXP2, MacCMD.MAC)

  io.out_ctrl := io.in_ctrl

  io.r_output.bits := Mux(ctrl.load_reg_li, reg, io.l_input.bits)
  io.r_output.valid := fire && (ctrl.load_reg_li || ctrl.flow_lr)

  io.d_output.bits := Mux(ctrl.mac && ctrl.acc_ui, macUnit.io.out_accType, io.u_input.bits)
  io.d_output.valid := fire && (ctrl.mac && ctrl.acc_ui || ctrl.flow_ud)

  io.u_output.bits := Mux(ctrl.mac && !ctrl.acc_ui, macUnit.io.out_accType, io.d_input.bits)
  io.u_output.valid := fire && (ctrl.mac && !ctrl.acc_ui || ctrl.flow_du)
}
