package msaga

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import msaga.sa._
import msaga.arithmetic._
import msaga.isa._
import msaga.utils.Ehr

object ConstIdx {
  def width = 1
  def ONE = 0
  def AttentionScale = 1
}

trait CanReadConstant {
  val is_constant = Bool()
}

class SpRead()(implicit p: Parameters) extends MSAGABundle with CanReadConstant {
  val addr = UInt(SPAD_ROW_ADDR_WIDTH.W)
  val rev_sram_out = Bool()
  val delay_sram_out = Bool()
  val rev_delayer_out = Bool()
}

class AccRead()(implicit p: Parameters) extends MSAGABundle with CanReadConstant {
  val addr = UInt(ACC_ROW_ADDR_WIDTH.W)
  val const_idx = UInt(ConstIdx.width.W)
  // read-modify-write, write back SRAM next cycle if set to 1
  val rmw = Bool()
}

/*
Pass `ArithmeticImpl` in because some execution plans may vary on it
*/
class MatrixEngineController[E <: Data : Arithmetic, A <: Data : Arithmetic](
  impl: ArithmeticImpl[E, A]
)(implicit p: Parameters) extends MSAGAModule {

  val io = IO(new Bundle {
    val in = Flipped(Decoupled(new MatrixInstruction(SPAD_ROW_ADDR_WIDTH, ACC_ROW_ADDR_WIDTH)))
    val sp_read = Valid(new SpRead)
    val acc_read = Valid(new AccRead())
    val cmp_ctrl = Valid(new CmpControl)
    val pe_ctrl = Vec(DIM, Valid(new PECtrl))
    val acc_ctrl = Valid(new AccumulatorControl)
  })

  val (planFunc, allPlans) = msagaParams.supportedExecutionPlans(DIM, impl).unzip

  val rs1 = RegEnable(io.in.bits.spad, io.in.fire)
  // we need to modify the content in the queue, so can not use chisel.util.Queue
  val rs2_queue = Reg(Vec(2, new MatrixInstructionAcc(ACC_ROW_ADDR_WIDTH)))
  val rs2_deq_ptr = RegInit(0.U(1.W))
  val rs2_enq_ptr = RegInit(0.U(1.W))
  val rs2 = rs2_queue(rs2_deq_ptr)

  val requireAccum = Cat(planFunc.zip(allPlans).map{ case (func, plan) =>
    func === io.in.bits.header.func && (plan.accumulateMaxCycle > 0).B
  }).orR
  when(io.in.fire && requireAccum) {
    rs2_queue(rs2_enq_ptr) := io.in.bits.acc
    rs2_enq_ptr := rs2_enq_ptr + 1.U
  }

  // Make valid a EHR to allow back-to-back execution
  val valid = Ehr(2, Bool(), Some(false.B))
  val computeFlags = allPlans.map(_ => RegInit(false.B))
  val accumFlags = allPlans.map(_ => RegInit(false.B))
  val computeTimer = RegInit(0.U(allPlans.map(_.computeMaxCycle).max.U.getWidth.W))
  val accumTimer = RegInit(0.U(allPlans.map(plan =>
    plan.accumulateMaxCycle - plan.accStartCycle
  ).max.U.getWidth.W))

  // PE Control
  computeFlags.zip(allPlans).map{ case (flag, plan) =>
    plan.genPECtrl(computeTimer, flag)
  }.transpose.map(row =>
    row.map(ctrl => ctrl.asUInt).reduce(_|_).asTypeOf(new PECtrl)
  ).zip(io.pe_ctrl).foreach{ case (generated, pe_ctrl) =>
    pe_ctrl.bits := generated
    pe_ctrl.valid := generated.asUInt.orR
  }

  // Scratchpad read
  val (spReadValid, spReadCtrl) = computeFlags.zip(allPlans).flatMap { case (flag, plan) =>
    plan.sp_read.map { desc =>
      (flag && desc.valid(computeTimer)) -> desc.toHardware(rs1, rs2)
    }
  }.unzip
  io.sp_read.valid := Cat(spReadValid).orR
  io.sp_read.bits := Mux1H(spReadValid, spReadCtrl)
  when(io.sp_read.fire && !io.sp_read.bits.is_constant) {
    // next row
    rs1.addr := (rs1.addr.zext + rs1.stride).asUInt
  }

  // Accum RAM read
  val (accReadValid, accReadCtrl) = accumFlags.zip(allPlans).flatMap{ case (flag, plan) =>
    plan.acc_read.map { desc =>
      (flag && desc.valid(accumTimer, base = plan.accStartCycle)) -> desc.toHardware(rs1, rs2)
    }
  }.unzip
  io.acc_read.valid := Cat(accReadValid).orR
  io.acc_read.bits := Mux1H(accReadValid, accReadCtrl)
  when(io.acc_read.fire && !io.acc_read.bits.is_constant) {
    rs2.addr := (rs2.addr.zext + rs2.stride).asUInt
  }

  // CMP Control
  val (cmpValid, cmpCtrl) = computeFlags.zip(allPlans).flatMap{ case (flag, plan) =>
    plan.cmp_ctrl.map{ desc =>
      (flag && desc.valid(computeTimer)) -> desc.toHardware(rs1, rs2)
    }
  }.unzip
  io.cmp_ctrl.valid := Cat(cmpValid).orR
  io.cmp_ctrl.bits := Mux1H(cmpValid, cmpCtrl)

  // ACCUMULATOR Control
  val (accValid, accCtrl) = accumFlags.zip(allPlans).flatMap{ case (flag, plan) =>
    plan.acc_ctrl.map{ desc =>
      (flag && desc.valid(accumTimer, base = plan.accStartCycle)) -> desc.toHardware(rs1, rs2)
    }
  }.unzip
  io.acc_ctrl.valid := Cat(accValid).orR
  io.acc_ctrl.bits := Mux1H(accValid, accCtrl)


  // Update flags / timers
  val (computeDone, accumDone) = computeFlags.zip(accumFlags).zip(allPlans).map{ case ((cf, af), plan) =>
    val cDone = cf && plan.computeDone(computeTimer)
    val aDone = af && plan.accumDone(accumTimer)
    val aStart = cf && {
      if (plan.accStartCycle > 0){ computeTimer === (plan.accStartCycle - 1).U }
      else false.B
    }
    when(cDone) {
      cf := false.B
    }
    when(aStart) {
      af := true.B
    }
    when(aDone) { af := false.B }
    (cDone, aDone)
  }.unzip

  when(Cat(computeDone).orR) {
    computeTimer := 0.U
    valid.write(0, false.B)
  }.elsewhen(valid.io.read(0)){
    computeTimer := computeTimer + 1.U
  }

  when(Cat(accumDone).orR) {
    accumTimer := 0.U
    rs2_deq_ptr := rs2_deq_ptr + 1.U
  }.elsewhen(Cat(accumFlags).orR) {
    accumTimer := accumTimer + 1.U
  }

  when(io.in.fire) {
    val set_cf = computeFlags.zip(accumFlags).zip(allPlans).zip(planFunc).map{ case (((cf, af), plan), func) =>
      val sel = func === io.in.bits.header.func
      if (plan.computeMaxCycle > 0) {
        cf := sel
      }
      if (plan.accumulateMaxCycle > 0 && plan.accStartCycle == 0) {
        af := sel
      }
      (plan.computeMaxCycle > 0).B && sel
    }
    valid.write(1, Cat(set_cf).orR)
  }
  val accReady = Cat(accumFlags) === 0.U ||
    Cat(accumDone).orR ||
    !io.in.bits.header.waitPrevAcc

  io.in.ready := !valid.read(1) && accReady

  when(RegNext(valid.read(0), false.B)) {
    assert(RegNext(PopCount(computeFlags) <= 1.U))
  }
  assert(PopCount(accumFlags) <= 1.U)
}
