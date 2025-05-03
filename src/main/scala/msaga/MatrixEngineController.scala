package msaga

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import msaga.sa._

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
}

class MatrixEngineController(implicit p: Parameters) extends MSAGAModule {

  val io = IO(new Bundle {
    val in = Flipped(Decoupled(new Instruction))
    val sp_read = Valid(new SpRead)
    val acc_read = Valid(new AccRead())
    val cmp_ctrl = Valid(new CmpControl)
    val pe_ctrl = Vec(DIM, Valid(new PECtrl))
    val acc_ctrl = Valid(new AccumulatorControl)
  })

  val rs1 = RegEnable(io.in.bits.rs1.asTypeOf(new MatrixRs(SPAD_ROW_ADDR_WIDTH)), io.in.fire)
  val rs2 = RegEnable(io.in.bits.rs2.asTypeOf(new MatrixRs(ACC_ROW_ADDR_WIDTH)), io.in.fire)

  val valid = RegInit(false.B)

  val (planFunc, allPlans) = msagaParams.supportedExecutionPlans(DIM).unzip
  for (pl <- allPlans) {
    println(pl.computeMaxCycle, pl.accumulateMaxCycle)
  }

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
    valid := false.B
  }.elsewhen(valid){
    computeTimer := computeTimer + 1.U
  }

  when(Cat(accumDone).orR) {
    accumTimer := 0.U
  }.elsewhen(Cat(accumFlags).orR) {
    accumTimer := accumTimer + 1.U
  }

  when(io.in.fire) {
    valid := true.B
    computeFlags.zip(accumFlags).zip(allPlans).zip(planFunc).foreach{ case (((cf, af), plan), func) =>
      val sel = func === io.in.bits.funct7
      cf := sel
      if (plan.accumulateMaxCycle > 0 && plan.accStartCycle == 0) {
        af := sel
      }
    }

  }
  io.in.ready := !valid

  when(valid) {
    assert(PopCount(computeFlags) === 1.U)
  }
  assert(PopCount(accumFlags) <= 1.U)
}
