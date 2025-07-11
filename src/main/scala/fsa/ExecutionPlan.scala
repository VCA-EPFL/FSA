package fsa

import collection.mutable.ListBuffer
import chisel3._
import chisel3.util._
import fsa.arithmetic.HasArithmeticParams
import fsa.isa._
import fsa.sa._
import fsa.utils.UIntRangeHelper._
import org.chipsalliance.cde.config.Parameters

trait CanGenerateHw[T <: Data] {
  def toHardware(rs1: MatrixInstructionSpad, rs2: MatrixInstructionAcc)(implicit p: Parameters): T
}

trait HasEffRange {
  val cycle: Int
  val repeat: Int

  def valid(t: UInt, base: Int = 0): Bool = {
    require(repeat >= 1)
    if (cycle < 0) false.B else t.between(cycle - base, cycle + repeat - base)
  }
}

trait ExecutionPlan {
  val rows: Int
  val cols: Int
  // PE control signals are more complex than spad/acc/cmp control, use a dedicate `ControlGen` to optimize them
  val pe_signals = (0 until 9).map(_ => ControlGen(rows)).toList
  val mac :: acc_ui :: load_reg_li :: load_reg_ui :: flow_lr :: flow_ud :: flow_du :: update_reg :: exp2 :: Nil = pe_signals

  def genPECtrl(timer: UInt, valid: Bool): Vec[PECtrl] = {
    val pe_ctrl = Wire(Vec(rows, new PECtrl))
    pe_signals
      .zip(pe_ctrl.map(_.getCtrlElements).transpose)
      .foreach { case (gen, ctrlBits) =>
        gen.generateCtrl(timer, valid).zip(ctrlBits).foreach {
          case (generated, connected) => connected := generated
        }
      }
    pe_ctrl
  }

  case class ConstRead(idx: Int, revIn: Boolean, revOut: Boolean, delay: Boolean)

  case class SpReadDesc(cycle: Int, repeat: Int, const: Option[ConstRead]) extends
    CanGenerateHw[SpRead] with HasEffRange
  {
    override def toHardware(rs1: MatrixInstructionSpad, rs2: MatrixInstructionAcc)(implicit p: Parameters): SpRead = {
      val r = Wire(new SpRead())
      r.rev_sram_out := rs1.revInput
      r.rev_delayer_out := rs1.revOutput
      r.delay_sram_out := rs1.delayOutput
      r.addr := rs1.addr
      r.is_constant := const.nonEmpty.B
      const.foreach { c =>
        r.rev_sram_out := c.revIn.B
        r.rev_delayer_out := c.revOut.B
        r.delay_sram_out := c.delay.B
        r.addr := c.idx.U
      }
      r
    }
  }

  case class AccReadDesc(cycle: Int, repeat: Int, const: Option[ConstRead], rmw: Boolean)
    extends CanGenerateHw[AccRead] with HasEffRange
  {
    override def toHardware(rs1: MatrixInstructionSpad, rs2: MatrixInstructionAcc)(implicit p: Parameters): AccRead = {
      val r = Wire(new AccRead())
      r.addr := rs2.addr
      r.is_constant := const.nonEmpty.B || rs2.zero
      r.const_idx := const.map(_.idx).getOrElse(AccConstIdx.ZERO).U
      r.rmw := rmw.B
      r
    }
  }

  case class CmpCtrlDesc(cycle: Int, repeat: Int, command: UInt)
    extends CanGenerateHw[CmpControl] with HasEffRange
  {
    override def toHardware(rs1: MatrixInstructionSpad, rs2: MatrixInstructionAcc)(implicit p: Parameters): CmpControl = {
      val ctrl = Wire(new CmpControl)
      ctrl.cmd := command
      ctrl
    }
  }

  case class AccCtrlDesc(cycle: Int, repeat: Int, command: UInt)
    extends CanGenerateHw[AccumulatorControl] with HasEffRange
  {
    override def toHardware(rs1: MatrixInstructionSpad, rs2: MatrixInstructionAcc)(implicit p: Parameters): AccumulatorControl = {
      val ctrl = Wire(new AccumulatorControl)
      ctrl.cmd := command
      ctrl
    }
  }

  def useAccTimer(cycle: Int): Boolean = {
    require(
      cycle < computeMaxCycle || cycle < accumulateMaxCycle,
      f"cycle $cycle must be less than computeMaxCycle $computeMaxCycle or accumulateMaxCycle $accumulateMaxCycle"
    )
    accumulateMaxCycle > 0 && cycle >= accStartCycle
  }

  val sp_read = ListBuffer[SpReadDesc]()
  val cmp_ctrl = ListBuffer[CmpCtrlDesc]()
  val acc_read = ListBuffer[AccReadDesc]()
  val acc_ctrl = ListBuffer[AccCtrlDesc]()

  var semaphoreReleaseCycle: Option[Int] = None
  var conflictFreeCycle: Option[Int] = None

  lazy val sem_write: HasEffRange = new HasEffRange {
    override val cycle: Int = semaphoreReleaseCycle.getOrElse(-1)
    override val repeat: Int = 1
  }
  lazy val conflict_free: HasEffRange = new HasEffRange {
    override val cycle: Int = conflictFreeCycle.get
    override val repeat: Int = 1
  }

  def readScratchPad(cycle: Int, repeat: Int, const: Option[ConstRead]) = {
    sp_read += SpReadDesc(cycle, repeat, const)
  }

  def readAccRAM(cycle: Int, repeat: Int, const: Option[ConstRead], rmw: Boolean = true) = {
    acc_read += AccReadDesc(cycle, repeat, const, rmw)
  }

  def setComparator(cycle: Int, repeat: Int, command: UInt) = {
    cmp_ctrl += CmpCtrlDesc(cycle, repeat, command)
  }

  def setAccumulator(cycle: Int, repeat: Int, command: UInt) = {
    acc_ctrl += AccCtrlDesc(cycle, repeat, command)
  }

  def releaseSemaphore(cycle: Int) = {
    require(semaphoreReleaseCycle.isEmpty)
    semaphoreReleaseCycle = Some(cycle)
  }

  def setConflictFree(cycle: Int) = {
    require(conflictFreeCycle.isEmpty)
    conflictFreeCycle = Some(cycle)
  }

  // exclusive
  def maxCycle(desc_list: Seq[HasEffRange]) = desc_list.map(d => d.cycle + d.repeat).maxOption.getOrElse(0)

  def computeMaxCycle = (
    Seq(sp_read, cmp_ctrl).map(x => maxCycle(x.toSeq)) ++ pe_signals.map(_.maxCycle)
  ).max

  // inclusive
  def accStartCycle = {
    Seq(acc_read, acc_ctrl).flatMap(_.map(_.cycle)).minOption.map { m =>
      Seq(m, computeMaxCycle).min
    }.getOrElse(-1)
  }

  def accumulateMaxCycle = Seq(acc_read, acc_ctrl).map(x => maxCycle(x.toSeq)).max

  def computeDone(timer: UInt) = if (computeMaxCycle == 0) true.B else timer === (computeMaxCycle - 1).U

  def accumDone(timer: UInt) = {
    if (accumulateMaxCycle > 0) {
      timer === (accumulateMaxCycle - 1 - accStartCycle).U
    } else {
      false.B
    }
  }

}

class LoadStationary(val rows: Int, val cols: Int) extends ExecutionPlan {
  // read Q from spad
  readScratchPad(0, cols, None)
  // release the semaphore immediately at the last cycle of reading sram
  releaseSemaphore(cols - 1)
  // load into systolic array
  load_reg_li.parallel(1, cols)
  /*
    Although we would occupy pe control signals until cycle `cols`,
    the next instruction should always read sram first (with 1 cycle
    latency), so we can start the next instruction at cycle `cols-1`
   */
  setConflictFree(cols - 1)
}

class AttentionScoreExecPlan(val rows: Int, val cols: Int, ap: HasArithmeticParams) extends ExecutionPlan {
  /****** S = Q @ K ******/
  // read K from spad
  readScratchPad(0, rows, None)
  // release the semaphore immediately at the last cycle of reading sram
  releaseSemaphore(rows - 1)
  // stream in K, multiply with Q from bottom left of the SA
  mac.flow_up(1, rows)
  flow_lr.flow_up(1, rows)

  /****** Put S back to systolic array ******/
  // the operations above takes `rows` cycles (the first element of
  // S = Q @ Kt reaches the upper left of the SA),
  // so we need to wait for `rows` cycles before we can read S from the SA
  flow_ud.flow_down(rows + 1, rows)
  // meanwhile, update the row max
  setComparator(rows + 1, rows, CmpControlCmd.UPDATE)

  /****** Flow zero bottom-up for later exp sum ******/
  flow_du.flow_up(rows + 4, rows)

  // prepare input for next cycle
  load_reg_ui.parallel(2 * rows + 1, 1)
  setComparator(2 * rows + 1, 1, CmpControlCmd.PROP_MAX)
  readScratchPad(
    2 * rows + 1, 1,
    Some(ConstRead(SpadConstIdx.ONE, revIn = false, revOut = false, delay = true))
  )
  /****** Staring from the first column, do element-wise ops ******/
  // s = s * 1 + (-m)
  update_reg.flow_down(2 * rows + 2, 1)
  acc_ui.flow_down(2 * rows + 2, 1)
  flow_ud.flow_down(2 * rows + 2, 1)
  flow_lr.flow_down(2 * rows + 2, 1)

  // prepare input for next cycle
  setComparator(2 * rows + 2, 1, CmpControlCmd.PROP_MAX_DIFF)
  readScratchPad(
    2 * rows + 2, 1,
    Some(ConstRead(SpadConstIdx.AttentionScale, revIn = false, revOut = false, delay = true))
  )
  // pass down delta_m; compute (s-m) * log2e in place
  flow_ud.flow_down(2 * rows + 3, 1)
  update_reg.flow_down(2 * rows + 3, 1)
  flow_lr.flow_down(2 * rows + 3, 1)

  val exp2_start = 2 * rows + 4
  val exp2_cycles = ap.exp2PwlPieces
  val exp2_end = exp2_start + exp2_cycles - 1

  setComparator(exp2_start - 1, exp2_cycles, CmpControlCmd.PROP_EXP2_INTERCEPTS)
  readScratchPad(
    exp2_start - 1, exp2_cycles,
    Some(ConstRead(SpadConstIdx.Exp2Slopes, revIn = false, revOut = false, delay = true))
  )
  // use pow2 to generate exp
  flow_ud.flow_down(exp2_start, exp2_cycles)
  flow_lr.flow_down(exp2_start, exp2_cycles)
  acc_ui.flow_down(exp2_start, exp2_cycles)
  exp2.flow_down(exp2_start, exp2_cycles)

  // prepare input for next cycle
  setComparator(exp2_end, 1, CmpControlCmd.PROP_ZERO)
  readScratchPad(
    exp2_end, 1,
    Some(ConstRead(SpadConstIdx.ONE, revIn = false, revOut = false, delay = true))
  )
  // use mac to compute the sum of exp
  mac.flow_down(exp2_end + 1, 1)
  acc_ui.flow_down(exp2_end + 1, 1)
  flow_lr.flow_down(exp2_end + 1, 1)

  /*
    The next instruction should always be `S @ V`,
    start it at cycle exp2_end + 1 so that the sram read
    of the next instruction can overlap with the last compute
    of the current instruction
  */
  setConflictFree(exp2_end)

  // collect diff = row_max(i-1) - row_max(i), and compute exp(diff)
  setAccumulator(2 * rows + rows + cols + 2, 1, AccumulatorCmd.EXP_S1)
  setAccumulator(2 * rows + rows + cols + 3, 1, AccumulatorCmd.EXP_S2)
  // update exp sum
  // readAccRAM(2 * rows + rows + cols + 3, 1, None)
  readAccRAM(exp2_end + rows + cols - 1, 1, None)
  // setAccumulator(2 * rows + rows + cols + 4, 1, AccumulatorCmd.ACC_SA)
  setAccumulator(exp2_end + rows + cols, 1, AccumulatorCmd.ACC_SA)
}

class AttentionValueExecPlan(val rows: Int, val cols: Int) extends ExecutionPlan {
  /****** O = P @ V ******/
  // read V from spad
  readScratchPad(0, rows, None)
  // release the semaphore immediately at the last cycle of reading sram
  releaseSemaphore(rows - 1)
  // V enters the SA from upper left
  mac.flow_down(1, rows)
  acc_ui.flow_down(1, rows)
  flow_lr.flow_down(1, rows)

  /*
    If the next instruction is load stationary (next inner loop start),
    the last compute of the current instructions happens at cycle `2*rows - 1`,
    we release at `2*rows - 2` to allow the sram read of the next instruction
    overlap with current compute
  */
  setConflictFree(2 * rows - 1 - 1)

  // read old O out from accumulator sram
  readAccRAM(rows + cols - 1, rows, None)
  // accumulate, update O
  setAccumulator(rows + cols, rows, AccumulatorCmd.ACC_SA)
}

// load one row from AccRAM to accumulator and get the reciprocal
class AttentionLseNormScale
(
  val rows: Int,
  val cols: Int,
  ap: HasArithmeticParams
) extends ExecutionPlan {
  readAccRAM(0, 1, None, rmw = false)
  setAccumulator(1, 1, AccumulatorCmd.SET_SCALE)
  setAccumulator(2, ap.reciprocalLatency, AccumulatorCmd.RECIPROCAL)
  releaseSemaphore(2 + ap.reciprocalLatency - 1)
  // This is a blocking instruction
  setConflictFree(2 + ap.reciprocalLatency - 1)
}

// perform the final lse norm after each flash attention inner loop
class AttentionLseNorm(val rows: Int, val cols: Int) extends ExecutionPlan {
  setComparator(0, 1, CmpControlCmd.RESET)
  readAccRAM(0, rows, None)
  setAccumulator(1, rows, AccumulatorCmd.ACC)
  releaseSemaphore(rows)
  // This is a blocking instruction
  setConflictFree(rows)
}