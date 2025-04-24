package msaga

import chisel3._
import chisel3.util._
import msaga.sa._
import ArithmeticSyntax._
import org.chipsalliance.cde.config.{Config, Field, Parameters}

case object MSAGAKey extends Field[MSAGAParams]

case class MSAGAParams(
  dim: Int,
  spadSizeBytes: Int,
  accSizeBytes: Int,
)

trait HasMSAGAParams {
  implicit val p: Parameters
  val msagaParams = p(MSAGAKey)
  def DIM = msagaParams.dim
  def DIM_WIDTH = log2Up(msagaParams.dim)

  def SPAD_ROWS = msagaParams.spadSizeBytes / (2*DIM)
  def SPAD_ROW_ADDR_WIDTH = log2Up(SPAD_ROWS)

  def ACC_ROWS = msagaParams.accSizeBytes / (2*DIM)
  def ACC_ROW_ADDR_WIDTH = log2Up(ACC_ROWS)

  def SRAM_ROW_ADDR_WIDTH = Seq(SPAD_ROW_ADDR_WIDTH, ACC_ROW_ADDR_WIDTH).max
}

abstract class MSAGABundle(implicit val p: Parameters) extends Bundle with HasMSAGAParams
abstract class MSAGAModule(implicit val p: Parameters) extends Module with HasMSAGAParams

class MSAGA[E <: Data : Arithmetic, A <: Data : Arithmetic](
  elemType: E, accType: A,
  mac: () => MacUnit[E, A],
  cmp: () => CmpUnit[A]
)(implicit p: Parameters) extends MSAGAModule {

  val io = IO(new Bundle {
    val inst = Flipped(Decoupled(new Instruction))
    val debug_sram_write = Input(new Bundle {
      val en = Bool()
      val addr = UInt(SPAD_ROW_ADDR_WIDTH.W)
      val data = Vec(DIM, elemType)
    })
  })

  val mxControl = Module(new MatrixEngineController)
  val inputDelayer = Module(new InputDelayer(DIM, elemType))
  val sa = Module(new SystolicArray(DIM, DIM, elemType, accType, mac, cmp))
  val sram = SRAM(
    SPAD_ROWS, Vec(DIM, elemType),
    numReadPorts = 1, numWritePorts = 1, numReadwritePorts = 0
  )

  dontTouch(mxControl.io)
  dontTouch(inputDelayer.io)
  dontTouch(sa.io)
  dontTouch(sram)

  mxControl.io.in <> io.inst

  sram.writePorts.head.enable := io.debug_sram_write.en
  sram.writePorts.head.address := io.debug_sram_write.addr
  sram.writePorts.head.data := io.debug_sram_write.data

  sram.readPorts.head.enable := mxControl.io.sp_read.valid && !mxControl.io.sp_read.bits.is_constant
  sram.readPorts.head.address := mxControl.io.sp_read.bits.addr

  val constSel = RegEnable(mxControl.io.sp_read.bits.addr(ConstGen.width - 1, 0), mxControl.io.sp_read.valid)
  val constVal = Mux(constSel === ConstGen.Lg2E, elemType.lg2_e, elemType.one)

  inputDelayer.io.in.valid := RegNext(mxControl.io.sp_read.valid, false.B)
  inputDelayer.io.in.bits.data := Mux(RegNext(mxControl.io.sp_read.bits.is_constant),
    VecInit(Seq.fill(DIM)(constVal)),
    sram.readPorts.head.data
  )
  inputDelayer.io.in.bits.rev_input := RegNext(mxControl.io.sp_read.bits.rev_sram_out)
  inputDelayer.io.in.bits.delay_output := RegNext(mxControl.io.sp_read.bits.delay_sram_out)
  inputDelayer.io.in.bits.rev_output := RegNext(mxControl.io.sp_read.bits.rev_delayer_out)

  sa.io.pe_data := inputDelayer.io.out
  sa.io.cmp_ctrl := mxControl.io.cmp_ctrl
  sa.io.pe_ctrl := mxControl.io.pe_ctrl
}