package msaga

import collection.mutable.ListBuffer
import chisel3._
import chisel3.util._
import msaga.arithmetic.HasArithmeticParams
import msaga.isa._
import msaga.sa._
import msaga.utils.UIntRangeHelper._
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
  val dim: Int
  // PE control signals are more complex than spad/acc/cmp control, use a dedicate `ControlGen` to optimize them
  val pe_signals = (0 until 9).map(_ => ControlGen(dim)).toList
  val mac :: acc_ui :: load_reg_li :: load_reg_ui :: flow_lr :: flow_ud :: flow_du :: update_reg :: exp2 :: Nil = pe_signals

  def genPECtrl(timer: UInt, valid: Bool): Vec[PECtrl] = {
    val pe_ctrl = Wire(Vec(dim, new PECtrl))
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

  case class SemaphoreWrite(cycle: Int) extends HasEffRange {
    override val repeat: Int = 1
    def useAccumTimer: Boolean = {
      require(cycle < computeMaxCycle || cycle < accumulateMaxCycle)
      accumulateMaxCycle > 0 && cycle >= accStartCycle
    }
  }

  val sp_read = ListBuffer[SpReadDesc]()
  val cmp_ctrl = ListBuffer[CmpCtrlDesc]()
  val acc_read = ListBuffer[AccReadDesc]()
  val acc_ctrl = ListBuffer[AccCtrlDesc]()
  var sem_write: SemaphoreWrite = SemaphoreWrite(-1)

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
    sem_write = SemaphoreWrite(cycle)
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

class LoadStationary(val dim: Int) extends ExecutionPlan {
  // read Q from spad
  readScratchPad(0, dim, None)
  // release the semaphore immediately at the last cycle of reading sram
  releaseSemaphore(dim - 1)
  // load into systolic array
  load_reg_li.parallel(1, dim)
}

class AttentionScoreExecPlan(val dim: Int) extends ExecutionPlan {
  /****** S = Q @ K ******/
  // read K from spad
  readScratchPad(0, dim, None)
  // release the semaphore immediately at the last cycle of reading sram
  releaseSemaphore(dim - 1)
  // stream in K, multiply with Q from bottom left of the SA
  mac.flow_up(1, dim)
  flow_lr.flow_up(1, dim)

  /****** Put S back to systolic array ******/
  flow_ud.flow_down(dim + 1, dim)
  // meanwhile, update the row max
  setComparator(dim + 1, dim, CmpControlCmd.UPDATE)

  /****** Flow zero bottom-up for later exp sum ******/
  flow_du.flow_up(dim + 4, dim)

  // prepare input for next cycle
  load_reg_ui.parallel(2 * dim + 1, 1)
  setComparator(2 * dim + 1, 1, CmpControlCmd.PROP_MAX)
  readScratchPad(
    2 * dim + 1, 1,
    Some(ConstRead(SpadConstIdx.ONE, revIn = false, revOut = false, delay = true))
  )
  /****** Staring from the first column, do element-wise ops ******/
  // s = s * 1 + (-m)
  update_reg.flow_down(2 * dim + 2, 1)
  acc_ui.flow_down(2 * dim + 2, 1)
  flow_ud.flow_down(2 * dim + 2, 1)
  flow_lr.flow_down(2 * dim + 2, 1)

  // prepare input for next cycle
  setComparator(2 * dim + 2, 1, CmpControlCmd.PROP_MAX_DIFF)
  readScratchPad(
    2 * dim + 2, 1,
    Some(ConstRead(SpadConstIdx.AttentionScale, revIn = false, revOut = false, delay = true))
  )
  // pass down delta_m; compute (s-m) * log2e in place
  flow_ud.flow_down(2 * dim + 3, 1)
  update_reg.flow_down(2 * dim + 3, 1) // acc_ui is false, not need to control
  flow_lr.flow_down(2 * dim + 3, 1)
  // use pow2 to generate exp
  exp2.flow_down(2 * dim + 4, 1)

  // prepare input for next cycle
  setComparator(2 * dim + 4, 1, CmpControlCmd.PROP_ZERO)
  readScratchPad(
    2 * dim + 4, 1,
    Some(ConstRead(SpadConstIdx.ONE, revIn = false, revOut = false, delay = true))
  )
  // use mac to compute the sum of exp
  mac.flow_down(2 * dim + 5, 1)
  acc_ui.flow_down(2 * dim + 5, 1)
  flow_lr.flow_down(2 * dim + 5, 1)

  // collect diff = row_max(i-1) - row_max(i), and compute exp(diff)
  setAccumulator(4 * dim + 3, 1, AccumulatorCmd.EXP_S1)
  setAccumulator(4 * dim + 4, 1, AccumulatorCmd.EXP_S2)
  // update exp sum
  readAccRAM(4 * dim + 4, 1, None)
  setAccumulator(4 * dim + 5, 1, AccumulatorCmd.ACC_SA)
}

class AttentionValueExecPlan(val dim: Int) extends ExecutionPlan {
  /****** O = P @ V ******/
  // read V from spad
  readScratchPad(0, dim, None)
  // release the semaphore immediately at the last cycle of reading sram
  releaseSemaphore(dim - 1)
  // V enters the SA from upper left
  mac.flow_down(1, dim)
  acc_ui.flow_down(1, dim)
  flow_lr.flow_down(1, dim)
  // read old O out from accumulator sram
  readAccRAM(2 * dim - 1, dim, None)
  // accumulate, update O
  setAccumulator(2 * dim, dim, AccumulatorCmd.ACC_SA)
}

// load one row from AccRAM to accumulator and get the reciprocal
class AttentionLseNormScale
(
  val dim: Int,
  ap: HasArithmeticParams
) extends ExecutionPlan {
  readAccRAM(0, 1, None, rmw = false)
  setAccumulator(1, 1, AccumulatorCmd.SET_SCALE)
  setAccumulator(2, ap.reciprocalLatency, AccumulatorCmd.RECIPROCAL)
  releaseSemaphore(2 + ap.reciprocalLatency - 1)
}

// perform the final lse norm after each flash attention inner loop
class AttentionLseNorm(val dim: Int) extends ExecutionPlan {
  setComparator(0, 1, CmpControlCmd.RESET)
  readAccRAM(0, dim, None)
  setAccumulator(1, dim, AccumulatorCmd.ACC)
  releaseSemaphore(dim)
}