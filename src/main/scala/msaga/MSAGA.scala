package msaga

import chisel3._
import chisel3.util._
import msaga.arithmetic.{Arithmetic, ArithmeticImpl, HasArithmeticParams}
import msaga.sa._
import msaga.arithmetic.ArithmeticSyntax._
import msaga.frontend.SemaphoreWrite
import msaga.isa.{ISA, MatrixInstruction}
import org.chipsalliance.cde.config.{Config, Field, Parameters}

case object MSAGAKey extends Field[Option[MSAGAParams]]

case class MSAGAParams(
  dim: Int,
  spadRows: Int,
  accRows: Int,
  instructionQueueEntries: Int = 256,
  dmaLoadInflight: Int = 16,
  dmaStoreInflight: Int = 8,
  // if not set, use the number of memory channels
  nMemPorts: Option[Int] = None,
  supportedExecutionPlans: (Int, HasArithmeticParams) => Seq[(UInt, ExecutionPlan)] = {
    (dim, ap) => Seq(
      ISA.MxFunc.LOAD_STATIONARY -> new LoadStationary(dim),
      ISA.MxFunc.ATTENTION_SCORE_COMPUTE -> new AttentionScoreExecPlan(dim),
      ISA.MxFunc.ATTENTION_VALUE_COMPUTE -> new AttentionValueExecPlan(dim),
      ISA.MxFunc.ATTENTION_LSE_NORM_SCALE -> new AttentionLseNormScale(dim, ap),
      ISA.MxFunc.ATTENTION_LSE_NORM -> new AttentionLseNorm(dim)
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
  def DIM = msagaParams.dim
  def DIM_WIDTH = log2Up(msagaParams.dim)

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
    val sem_write = Valid(new SemaphoreWrite)
    val debug_sram_io = new DebugSRAMIO(DIM)
    val debug_mx_inst = if (msagaParams.unitTestBuild) Some(Flipped(Decoupled(UInt(96.W)))) else None
  })

  val mxControl = Module(new MatrixEngineController(ev))
  val inputDelayer = Module(new InputDelayer(DIM, arithmeticImpl.elemType))
  val outputDelayer = Module(new OutputDelayer(DIM, arithmeticImpl.accType))
  val sa = Module(new SystolicArray[E, A](DIM, DIM))
  val accumulator = Module(new Accumulator[A](DIM, DIM, ev.accType, ev.accUnit _))
  val spRAM = SRAM(
    SPAD_ROWS, Vec(DIM, ev.elemType),
    numReadPorts = 2, numWritePorts = 2, numReadwritePorts = 0
  )
  val accRAM = SRAM(
    ACC_ROWS, Vec(DIM, ev.accType),
    numReadPorts = 2, numWritePorts = 2, numReadwritePorts = 0
  )

  io.debug_mx_inst.map { inst =>
    dontTouch(mxControl.io)
    dontTouch(inputDelayer.io)
    dontTouch(outputDelayer.io)
    dontTouch(sa.io)
    dontTouch(accumulator.io)
    io.inst.ready := false.B
    mxControl.io.in.valid := inst.valid
    mxControl.io.in.bits := inst.bits.asTypeOf(mxControl.io.in.bits)
    inst.ready := mxControl.io.in.ready
  }.getOrElse({
    mxControl.io.in <> io.inst
  })
  io.sem_write := mxControl.io.sem_write


  // TODO: connect with DMA
  spRAM.writePorts.head.enable := false.B
  spRAM.writePorts.head.address := DontCare
  spRAM.writePorts.head.data := DontCare


  spRAM.writePorts.last.enable := io.debug_sram_io.en_sp && io.debug_sram_io.write
  spRAM.writePorts.last.address := io.debug_sram_io.addr
  spRAM.writePorts.last.data.zip(io.debug_sram_io.wdata).foreach { case (a, b) => a := b.asTypeOf(a)}

  accRAM.writePorts.last.enable := io.debug_sram_io.en_acc && io.debug_sram_io.write
  accRAM.writePorts.last.address := io.debug_sram_io.addr
  accRAM.writePorts.last.data.zip(io.debug_sram_io.wdata).foreach{ case (a, b) => a := b.asTypeOf(a)}

  spRAM.readPorts.last.enable := io.debug_sram_io.en_sp && io.debug_sram_io.read
  spRAM.readPorts.last.address := io.debug_sram_io.addr

  accRAM.readPorts.last.enable := io.debug_sram_io.en_acc && io.debug_sram_io.read
  accRAM.readPorts.last.address := io.debug_sram_io.addr

  io.debug_sram_io.rdata.zipWithIndex.foreach{ case (io_rd, i) =>
    io_rd := Mux(RegNext(io.debug_sram_io.en_sp),
      spRAM.readPorts.last.data(i).asUInt,
      accRAM.readPorts.last.data(i).asUInt
    )
  }

  spRAM.readPorts.head.enable := mxControl.io.sp_read.valid && !mxControl.io.sp_read.bits.is_constant
  spRAM.readPorts.head.address := mxControl.io.sp_read.bits.addr

  accRAM.readPorts.head.enable := mxControl.io.acc_read.valid && !mxControl.io.acc_read.bits.is_constant
  accRAM.readPorts.head.address := mxControl.io.acc_read.bits.addr


  val spConstList = VecInit(ev.elemType.one, ev.elemType.attentionScale(msagaParams.dim))
  val spConstSel = RegEnable(
    mxControl.io.sp_read.bits.addr(ConstIdx.width - 1, 0),
    mxControl.io.sp_read.valid && mxControl.io.sp_read.bits.is_constant
  )
  val spConstVal = spConstList(spConstSel)
  val accConstSel = RegEnable(
    mxControl.io.acc_read.bits.const_idx,
    mxControl.io.acc_read.valid && mxControl.io.acc_read.bits.is_constant
  )
  val accConstList = VecInit(ev.accType.one, ev.accType.attentionScale(msagaParams.dim))
  val accConstVal = accConstList(accConstSel)

  inputDelayer.io.in.valid := RegNext(mxControl.io.sp_read.valid, false.B)
  inputDelayer.io.in.bits.data := Mux(RegNext(mxControl.io.sp_read.bits.is_constant),
    VecInit(Seq.fill(DIM)(spConstVal)),
    spRAM.readPorts.head.data
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
    VecInit(Seq.fill(DIM)(accConstVal)),
    accRAM.readPorts.head.data
  )
  accumulator.io.ctrl_in := mxControl.io.acc_ctrl

  accRAM.writePorts.head.enable := RegNext(mxControl.io.acc_read.valid && mxControl.io.acc_read.bits.rmw, false.B)
  accRAM.writePorts.head.address := RegNext(mxControl.io.acc_read.bits.addr)
  accRAM.writePorts.head.data := accumulator.io.sram_out
}