package fsa

import chisel3._
import chisel3.util._
import fsa.arithmetic.{Arithmetic, ArithmeticImpl, HasArithmeticParams}
import fsa.sa._
import fsa.arithmetic.ArithmeticSyntax._
import fsa.frontend.Semaphore
import fsa.isa.{ISA, MatrixInstruction}
import fsa.utils.DelayedAssert
import org.chipsalliance.cde.config.{Config, Field, Parameters}
import fsa.arithmetic.FloatPoint

case object FSA extends Field[Option[FSAParams]]
case object FpFSAImplKey extends Field[Option[ArithmeticImpl[FloatPoint, FloatPoint]]]

case class FSAParams(
  saRows: Int,
  saCols: Int,
  spadRows: Int,
  accRows: Int,
  spadBanks: Int = 2,
  accBanks: Int = 2,
  instructionQueueEntries: Int = 256,
  mxInflight: Int = 8,
  dmaLoadInflight: Int = 16,
  dmaStoreInflight: Int = 8,
  nMemPorts: Int = 1,
  supportedExecutionPlans: (Int, Int, HasArithmeticParams) => Seq[(UInt, ExecutionPlan)] = {
    (rows, cols, ap) => Seq(
      ISA.MxFunc.LOAD_STATIONARY -> new LoadStationary(rows, cols),
      ISA.MxFunc.ATTENTION_SCORE_COMPUTE -> new AttentionScoreExecPlan(rows, cols, ap),
      ISA.MxFunc.ATTENTION_VALUE_COMPUTE -> new AttentionValueExecPlan(rows, cols),
      ISA.MxFunc.ATTENTION_LSE_NORM_SCALE -> new AttentionLseNormScale(rows, cols, ap),
      ISA.MxFunc.ATTENTION_LSE_NORM -> new AttentionLseNorm(rows, cols)
    )
  },
  unitTestBuild: Boolean = false
) {
  def spadAddrWidth = log2Up(spadRows)
  def accAddrWidth = log2Up(accRows)
  def sramAddrWidth = Seq(spadAddrWidth, accAddrWidth).max
  def dmaMaxInflight = Seq(dmaLoadInflight, dmaStoreInflight).max
}

trait HasFSAParams {
  implicit val p: Parameters
  val fsaParams = p(FSA).get
  def SA_ROWS = fsaParams.saRows
  def SA_COLS = fsaParams.saCols

  def SPAD_ROWS = fsaParams.spadRows
  def SPAD_ROW_ADDR_WIDTH = fsaParams.spadAddrWidth

  def ACC_ROWS = fsaParams.accRows
  def ACC_ROW_ADDR_WIDTH = fsaParams.accAddrWidth

  def SRAM_ROW_ADDR_WIDTH = fsaParams.sramAddrWidth
}

abstract class FSABundle(implicit val p: Parameters) extends Bundle with HasFSAParams
abstract class FSAModule(implicit val p: Parameters) extends Module with HasFSAParams

class FSA[E <: Data : Arithmetic, A <: Data : Arithmetic]
(
  arithmeticImpl: ArithmeticImpl[E, A],
  beatBytes: Int
)(implicit p: Parameters) extends FSAModule {

  implicit val ev: ArithmeticImpl[E, A] = arithmeticImpl

  val io = IO(new Bundle {
    val inst = Flipped(Decoupled(new MatrixInstruction(SPAD_ROW_ADDR_WIDTH, ACC_ROW_ADDR_WIDTH)))
    val sem_release = Valid(new Semaphore)
    // dma write spad
    val spad_write = Vec(fsaParams.nMemPorts, Flipped(new SRAMNarrowWrite(
      SPAD_ROW_ADDR_WIDTH, ev.elemType.getWidth, SA_ROWS, beatBytes
    )))
    // dma read accumulator
    val acc_read = Vec(fsaParams.nMemPorts, Flipped(new SRAMNarrowRead(
      ACC_ROW_ADDR_WIDTH, ev.accType.getWidth, SA_COLS, beatBytes
    )))
    val busy = Output(Bool())
  })

  val mxControl = Module(new MatrixEngineController(ev))
  val inputDelayer = Module(new InputDelayer(SA_ROWS, arithmeticImpl.elemType))
  val outputDelayer = Module(new OutputDelayer(SA_COLS, arithmeticImpl.accType))
  val sa = Module(new SystolicArray[E, A](SA_ROWS, SA_COLS))
  val accumulator = Module(new Accumulator[A](SA_ROWS, SA_COLS, ev.accType, ev.accUnit _))

  val spRAM = {
    val sram = Module(new BankedSRAM(
      SPAD_ROWS, SA_ROWS, ev.elemType.getWidth,
      fsaParams.spadBanks, beatBytes,
      nFullRead = 1, nFullWrite = 0,
      nNarrowRead = 0, nNarrowWrite = fsaParams.nMemPorts,
      moduleName = "ScratchPadSRAM"
    ))
    sram.io
  }

  val accRAM = {
    val sram = Module(new BankedSRAM(
      ACC_ROWS, SA_COLS, ev.accType.getWidth,
      fsaParams.accBanks, beatBytes,
      nFullRead = 1, nFullWrite = 1,
      nNarrowRead = fsaParams.nMemPorts, nNarrowWrite = 0,
      moduleName = "AccumulationSRAM"
    ))
    sram.io
  }

  Seq(spRAM.fullRead.head, spRAM.narrowWrite.head, accRAM.fullRead.head, accRAM.fullWrite.head).foreach { x =>
    DelayedAssert(!x.valid || x.ready)
  }

  mxControl.io.in <> io.inst
  io.sem_release := mxControl.io.sem_release
  io.busy := mxControl.io.busy

  spRAM.narrowWrite.zip(io.spad_write).foreach{ case (l, r) => l <> r }

  spRAM.fullRead.head.valid := mxControl.io.sp_read.valid && !mxControl.io.sp_read.bits.is_constant
  spRAM.fullRead.head.addr := mxControl.io.sp_read.bits.addr
  spRAM.fullRead.head.setFullMask()

  accRAM.fullRead.head.valid := mxControl.io.acc_read.valid && !mxControl.io.acc_read.bits.is_constant
  accRAM.fullRead.head.addr := mxControl.io.acc_read.bits.addr
  accRAM.fullRead.head.setFullMask()

  accRAM.narrowRead.zip(io.acc_read).foreach{ case (l, r) => l <> r }


  val exp2PwlSlopes = VecInit(ev.exp2PwlSlopes)
  val exp2PwlCounter = Counter(exp2PwlSlopes.length)
  val exp2PwlSlopeVal = exp2PwlSlopes(exp2PwlCounter.value)

  val spConstList = VecInit(ev.elemType.one, ev.elemType.attentionScale(SA_ROWS), exp2PwlSlopeVal)
  val spConstSel = RegEnable(
    mxControl.io.sp_read.bits.addr.take(SpadConstIdx.width),
    mxControl.io.sp_read.valid && mxControl.io.sp_read.bits.is_constant
  )
  val spConstVal = spConstList(spConstSel)

  when(RegNext(
    mxControl.io.sp_read.valid &&
    mxControl.io.sp_read.bits.is_constant &&
    mxControl.io.sp_read.bits.addr.take(SpadConstIdx.width) === SpadConstIdx.Exp2Slopes.U,
    false.B
  )) {
    exp2PwlCounter.inc()
  }

  val accConstSel = RegEnable(
    mxControl.io.acc_read.bits.const_idx,
    mxControl.io.acc_read.valid && mxControl.io.acc_read.bits.is_constant
  )
  val accConstList = VecInit(ev.accType.zero)
  val accConstVal = accConstList(accConstSel)

  inputDelayer.io.in.valid := RegNext(mxControl.io.sp_read.valid, false.B)
  inputDelayer.io.in.bits.data := Mux(RegNext(mxControl.io.sp_read.bits.is_constant),
    VecInit(Seq.fill(SA_ROWS)(spConstVal)),
    spRAM.fullRead.head.data.asTypeOf(inputDelayer.io.in.bits.data)
  )
  inputDelayer.io.in.bits.rev_input := RegNext(mxControl.io.sp_read.bits.rev_sram_out)
  inputDelayer.io.in.bits.delay_output := RegNext(mxControl.io.sp_read.bits.delay_sram_out)
  inputDelayer.io.in.bits.rev_output := RegNext(mxControl.io.sp_read.bits.rev_delayer_out)

  sa.io.pe_data := inputDelayer.io.out
  sa.io.cmp_ctrl := mxControl.io.cmp_ctrl
  sa.io.pe_ctrl := mxControl.io.pe_ctrl

  outputDelayer.io.in := VecInit(sa.io.acc_out.map(_.bits))

  accumulator.io.sa_in := outputDelayer.io.out
  accumulator.io.sram_in := Mux(RegNext(mxControl.io.acc_read.bits.is_constant),
    VecInit(Seq.fill(SA_COLS)(accConstVal)),
    accRAM.fullRead.head.data.asTypeOf(accumulator.io.sram_in)
  )
  accumulator.io.ctrl_in := mxControl.io.acc_ctrl

  accRAM.fullWrite.head.valid := RegNext(mxControl.io.acc_read.valid && mxControl.io.acc_read.bits.rmw, false.B)
  accRAM.fullWrite.head.addr := RegNext(mxControl.io.acc_read.bits.addr)
  accRAM.fullWrite.head.data := accumulator.io.sram_out.asTypeOf(accRAM.fullWrite.head.data)
  accRAM.fullWrite.head.setFullMask()
}