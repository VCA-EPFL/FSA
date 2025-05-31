package msaga

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import msaga.sa._
import msaga.arithmetic._
import msaga.frontend.Semaphore
import msaga.isa._
import msaga.utils.{DelayedAssert, Ehr}
import chisel3.experimental.SourceInfo

object SpadConstIdx {
  def width = 2
  def ONE = 0
  def AttentionScale = 1
  def Exp2Slopes = 2
}

object AccConstIdx {
  def width = 1
  def ZERO = 0
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
  val const_idx = UInt(AccConstIdx.width.W)
  // read-modify-write, write back SRAM next cycle if set to 1
  val rmw = Bool()
}


class MatrixControllerIO(implicit p: Parameters) extends MSAGABundle {
  val in = Flipped(Decoupled(new MatrixInstruction(SPAD_ROW_ADDR_WIDTH, ACC_ROW_ADDR_WIDTH)))
  val sp_read = Valid(new SpRead)
  val acc_read = Valid(new AccRead())
  val cmp_ctrl = Valid(new CmpControl)
  val pe_ctrl = Vec(SA_ROWS, Valid(new PECtrl))
  val acc_ctrl = Valid(new AccumulatorControl)
  val sem_release = Valid(new Semaphore)
  val busy = Output(Bool())
}

class MatrixControlFSM
(
  planFunc: Seq[UInt],
  executionPlans: Seq[ExecutionPlan]
)(implicit p: Parameters) extends MSAGAModule {
  val io = IO(new MatrixControllerIO() {
    val conflictFree = Output(Bool())
  })

  val header = RegEnable(io.in.bits.header, io.in.fire)
  val rs1 = RegEnable(io.in.bits.spad, io.in.fire)
  val rs2 = RegEnable(io.in.bits.acc, io.in.fire)

  val conflictFreeFlag = Ehr(2, Bool(), Some(false.B))
  val computeFlags = executionPlans.map(_ => RegInit(false.B))
  val accumFlags = executionPlans.map(_ => RegInit(false.B))

  val computeValid = Cat(computeFlags).orR
  val accumValid = Cat(accumFlags).orR

  val computeTimer = RegInit(0.U(executionPlans.map(_.computeMaxCycle).max.U.getWidth.W))
  // use a separate timer for accumulation to reduce comparision latency
  val accumTimer = RegInit(0.U(executionPlans.map(plan =>
    plan.accumulateMaxCycle - plan.accStartCycle
  ).max.U.getWidth.W))

  // PE Control
  computeFlags.zip(executionPlans).map{ case (flag, plan) =>
    plan.genPECtrl(computeTimer, flag)
  }.transpose.map(row =>
    row.map(ctrl => ctrl.asUInt).reduce(_|_).asTypeOf(new PECtrl)
  ).zip(io.pe_ctrl).foreach{ case (generated, pe_ctrl) =>
    pe_ctrl.bits := generated
    pe_ctrl.valid := generated.asUInt.orR
  }


  def select[T <: Data]
  (
    timer: UInt,
    flags: Seq[Bool], ctrlDesc: Seq[Iterable[CanGenerateHw[T] with HasEffRange]],
    timerBaseOpt: Option[Seq[Int]] = None): Valid[T] =
  {
    val timerBase = timerBaseOpt.getOrElse(flags.map(_ => 0))
    val (valid, ctrl) = flags.zip(ctrlDesc).zip(timerBase).flatMap { case ((flag, descSeq), tBase) =>
      descSeq.map{ desc =>
        (flag && desc.valid(timer, tBase)) -> desc.toHardware(rs1, rs2)
      }
    }.unzip
    val out = Wire(Valid(chiselTypeOf(ctrl.head)))
    out.valid := Cat(valid).orR
    out.bits := Mux1H(valid, ctrl)
    out
  }

  // Scratchpad read
  io.sp_read := select(computeTimer, computeFlags, executionPlans.map(_.sp_read))
  when(io.sp_read.fire) {
    // next row
    rs1.addr := (rs1.addr.zext + rs1.stride).asUInt
  }

  // Accum RAM read
  io.acc_read := select(accumTimer, accumFlags, executionPlans.map(_.acc_read), Some(executionPlans.map(_.accStartCycle)))
  when(io.acc_read.fire) {
    rs2.addr := (rs2.addr.zext + rs2.stride).asUInt
  }

  // CMP Control
  io.cmp_ctrl := select(computeTimer, computeFlags, executionPlans.map(_.cmp_ctrl))

  // ACCUMULATOR Control
  io.acc_ctrl := select(accumTimer, accumFlags, executionPlans.map(_.acc_ctrl), Some(executionPlans.map(_.accStartCycle)))

  // Release Semaphore
  io.sem_release.valid := Cat(computeFlags.zip(accumFlags).zip(executionPlans).map{ case ((cf, af), plan) =>
    (plan.sem_write.cycle > 0).B && (if (plan.useAccTimer(plan.sem_write.cycle)) {
      af && plan.sem_write.valid(accumTimer, base = plan.accStartCycle) && header.releaseValid
    } else {
      cf && plan.sem_write.valid(computeTimer) && header.releaseValid
    })
  }).orR
  io.sem_release.bits.id := header.semId
  io.sem_release.bits.value := header.releaseSemValue

  // Update flags / timers
  val (computeDone, conflictFree, accumDone) = computeFlags.zip(accumFlags).zip(executionPlans).map{ case ((cf, af), plan) =>
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
    val conflictFree = if (plan.useAccTimer(plan.conflict_free.cycle)) {
      af && plan.conflict_free.valid(accumTimer, base = plan.accStartCycle)
    } else {
      cf && plan.conflict_free.valid(computeTimer)
    }
    (cDone, conflictFree, aDone)
  }.unzip3

  when(Cat(computeDone).orR) {
    computeTimer := 0.U
  }.elsewhen(Cat(computeFlags).orR){
    computeTimer := computeTimer + 1.U
  }

  when(Cat(accumDone).orR) {
    accumTimer := 0.U
  }.elsewhen(Cat(accumFlags).orR) {
    accumTimer := accumTimer + 1.U
  }

  when(Cat(conflictFree).orR) {
    conflictFreeFlag.write(0, true.B)
  }

  when(io.in.fire) {
    val set_cf = computeFlags.zip(accumFlags).zip(executionPlans).zip(planFunc).map{ case (((cf, af), plan), func) =>
      val sel = func === io.in.bits.header.func
      if (plan.computeMaxCycle > 0) {
        cf := sel
      }
      if (plan.accumulateMaxCycle > 0 && plan.accStartCycle == 0) {
        af := sel
      }
      (plan.conflict_free.cycle == 0).B
    }
    conflictFreeFlag.write(1, Cat(set_cf).orR)
  }

  io.busy := computeValid || accumValid
  io.in.ready := !io.busy
  io.conflictFree := conflictFreeFlag.read(1)

  DelayedAssert(PopCount(computeFlags) <= 1.U)
  DelayedAssert(PopCount(accumFlags) <= 1.U)
}

/*
Pass `ArithmeticImpl` in because some execution plans may vary on it
*/
class MatrixEngineController[E <: Data : Arithmetic, A <: Data : Arithmetic](
  impl: ArithmeticImpl[E, A]
)(implicit p: Parameters) extends MSAGAModule {
  val io = IO(new MatrixControllerIO())

  /*
    we allow two instructions to be overlapped, they can control different
    part of the systolic array simultaneously
   */
  val fsm_list = (0 until 2) map { _ =>
    val (planFunc, allPlans) = msagaParams.supportedExecutionPlans(SA_ROWS, SA_COLS, impl).unzip
    val fsm = Module(new MatrixControlFSM(planFunc, allPlans))
    fsm
  }

  def Mux1HValidIO[T <: Data](in: Seq[Valid[T]], out: Valid[T]): Unit = {
    DelayedAssert(PopCount(in.map(_.valid)) <= 1.U)
    out.valid := Cat(in.map(_.valid)).orR
    out.bits := Mux1H(in.map(_.valid), in.map(_.bits))
  }

  val fsm_io = VecInit(fsm_list.map(_.io))
  Mux1HValidIO(fsm_io.map(_.sp_read), io.sp_read)
  Mux1HValidIO(fsm_io.map(_.acc_read), io.acc_read)
  Mux1HValidIO(fsm_io.map(_.cmp_ctrl), io.cmp_ctrl)
  Mux1HValidIO(fsm_io.map(_.acc_ctrl), io.acc_ctrl)
  Mux1HValidIO(fsm_io.map(_.sem_release), io.sem_release)
  io.pe_ctrl.zipWithIndex.foreach { case (out, idx) =>
    Mux1HValidIO(fsm_io.map(_.pe_ctrl(idx)), out)
  }

  val enq_ptr = RegInit(0.U(1.W))
  val deq_ptr = enq_ptr + 1.U
  when(io.in.fire) {
    enq_ptr := deq_ptr
  }

  val canEnq = Mux(io.in.bits.header.waitPrevAcc,
    !io.busy,
    !fsm_io(deq_ptr).busy || fsm_io(deq_ptr).conflictFree
  )
  io.in.ready := fsm_io(enq_ptr).in.ready && canEnq
  fsm_io.map(_.in).zipWithIndex.foreach{ case (in, idx) =>
    in.valid := io.in.valid && idx.U === enq_ptr && canEnq
    in.bits := io.in.bits
  }

  io.busy := Cat(fsm_io.map(_.busy)).orR
}
