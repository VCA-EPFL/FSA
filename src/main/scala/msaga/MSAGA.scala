package msaga

import chisel3._
import chisel3.util._
import msaga.arithmetic.{Arithmetic, ArithmeticImpl, HasArithmeticParams}
import msaga.sa._
import msaga.arithmetic.ArithmeticSyntax._
import msaga.frontend.Semaphore
import msaga.isa.{ISA, MatrixInstruction}
import msaga.utils.DelayedAssert
import org.chipsalliance.cde.config.{Config, Field, Parameters}
import msaga.arithmetic.FloatPoint

case object MSAGAKey extends Field[Option[MSAGAParams]]
case object FpMSAGAImplKey extends Field[Option[ArithmeticImpl[FloatPoint, FloatPoint]]]

case class MSAGAParams(
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

trait HasMSAGAParams {
  implicit val p: Parameters
  val msagaParams = p(MSAGAKey).get
  def SA_ROWS = msagaParams.saRows
  def SA_COLS = msagaParams.saCols

  def SPAD_ROWS = msagaParams.spadRows
  def SPAD_ROW_ADDR_WIDTH = msagaParams.spadAddrWidth

  def ACC_ROWS = msagaParams.accRows
  def ACC_ROW_ADDR_WIDTH = msagaParams.accAddrWidth

  def SRAM_ROW_ADDR_WIDTH = msagaParams.sramAddrWidth
}

abstract class MSAGABundle(implicit val p: Parameters) extends Bundle with HasMSAGAParams
abstract class MSAGAModule(implicit val p: Parameters) extends Module with HasMSAGAParams

// allow SRAMs read/write by external driver
class DebugSRAMIO(dim: Int) extends Bundle {
  val en_sp = Input(Bool())
  val en_acc = Input(Bool())
  val write = Input(Bool())
  val read = Input(Bool())
  val addr = Input(UInt(64.W))
  val wdata = Input(Vec(dim, UInt(64.W)))
  val rdata = Output(Vec(dim, UInt(64.W)))
}

class MSAGA[E <: Data : Arithmetic, A <: Data : Arithmetic]
(
  arithmeticImpl: ArithmeticImpl[E, A]
)(implicit p: Parameters) extends MSAGAModule {

  implicit val ev: ArithmeticImpl[E, A] = arithmeticImpl

  val io = IO(new Bundle {
    val inst = Flipped(Decoupled(new MatrixInstruction(SPAD_ROW_ADDR_WIDTH, ACC_ROW_ADDR_WIDTH)))
    val sem_release = Valid(new Semaphore)
    // dma write spad
    val spad_write = Vec(msagaParams.nMemPorts, Flipped(new SRAMWrite(SPAD_ROW_ADDR_WIDTH, ev.elemType, SA_ROWS)))
    // dma read accumulator
    val acc_read = Vec(msagaParams.nMemPorts, Flipped(new SRAMRead(ACC_ROW_ADDR_WIDTH, ev.accType, SA_COLS)))
    val busy = Output(Bool())
  })

  val mxControl = Module(new MatrixEngineController(ev))
  val inputDelayer = Module(new InputDelayer(SA_ROWS, arithmeticImpl.elemType))
  val outputDelayer = Module(new OutputDelayer(SA_COLS, arithmeticImpl.accType))
  val sa = Module(new SystolicArray[E, A](SA_ROWS, SA_COLS))
  val accumulator = Module(new Accumulator[A](SA_ROWS, SA_COLS, ev.accType, ev.accUnit _))

  val spRAM = Module(new BankedSRAM(
    SPAD_ROWS, ev.elemType, SA_ROWS,
    nBanks = msagaParams.spadBanks, nReadPorts = 1, nWritePorts = msagaParams.nMemPorts
  )).io

  /*
   * read port 0: matrix engine
   * read port [1, 1 + nMemPorts): dma
   * write port 0: matrix engine
   */
  val accRAM = Module(new BankedSRAM(
    ACC_ROWS, ev.accType, SA_COLS,
    nBanks = msagaParams.accBanks, nReadPorts = 1 + msagaParams.nMemPorts, nWritePorts = 1
  )).io

  Seq(spRAM.read.head, spRAM.write.head, accRAM.read.head, accRAM.write.head).foreach { x =>
    DelayedAssert(!x.valid || x.ready)
  }

  mxControl.io.in <> io.inst
  io.sem_release := mxControl.io.sem_release
  io.busy := mxControl.io.busy

  spRAM.write.zip(io.spad_write).foreach{ case (l, r) => l <> r }

  spRAM.read.head.valid := mxControl.io.sp_read.valid && !mxControl.io.sp_read.bits.is_constant
  spRAM.read.head.addr := mxControl.io.sp_read.bits.addr

  accRAM.read.head.valid := mxControl.io.acc_read.valid && !mxControl.io.acc_read.bits.is_constant
  accRAM.read.head.addr := mxControl.io.acc_read.bits.addr

  accRAM.read.tail.zip(io.acc_read).foreach{ case (l, r) => l <> r }


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
    spRAM.read.head.data
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
    accRAM.read.head.data
  )
  accumulator.io.ctrl_in := mxControl.io.acc_ctrl

  accRAM.write.head.valid := RegNext(mxControl.io.acc_read.valid && mxControl.io.acc_read.bits.rmw, false.B)
  accRAM.write.head.addr := RegNext(mxControl.io.acc_read.bits.addr)
  accRAM.write.head.data := accumulator.io.sram_out
  accRAM.write.head.mask := VecInit(Seq.fill(SA_COLS){ true.B })
}