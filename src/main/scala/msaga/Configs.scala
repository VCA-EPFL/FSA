package msaga

import org.chipsalliance.diplomacy.lazymodule._
import org.chipsalliance.cde.config._
import testchipip.soc.{SubsystemInjector, SubsystemInjectorKey}
import freechips.rocketchip.subsystem._
import freechips.rocketchip.amba.axi4._
import freechips.rocketchip.tilelink._
import msaga.arithmetic._
import chisel3._

case class MSAGAInjector[E <: Data : Arithmetic, A <: Data : Arithmetic](arithmeticImpl: ArithmeticImpl[E, A])
  extends SubsystemInjector((p, baseSubsystem) => {
  implicit val q = p
  val fbus = baseSubsystem.locateTLBusWrapper(FBUS)
  val domain = fbus.generateSynchronousDomain("msaga")
  domain {
    val msaga = LazyModule(new AXI4MSAGA(arithmeticImpl))
    fbus.coupleTo("msaga"){
      // AXI4Deinterleaver is not needed since msaga never generate interleaved resp
      msaga.configNode :=
        AXI4UserYanker() :=
        TLToAXI4() :=
        TLFragmenter(msaga.instBeatBytes, fbus.blockBytes, holdFirstDeny = true) :=
        TLWidthWidget(fbus.beatBytes) :=
        _
    }
  }
})

class WithFpMSAGA
(
  params: MSAGAParams = Configs.smallMSAGAParams,
  arithmeticImpl: ArithmeticImpl[FloatPoint, FloatPoint] = Configs.fp16MulFp32AddArithmeticImpl
) extends Config((site, here, up) => {
  case MSAGAKey => Some(params)
  case SubsystemInjectorKey => up(SubsystemInjectorKey) + MSAGAInjector(arithmeticImpl)
})

object Configs {
  lazy val smallMSAGAParams = MSAGAParams(
    dim = 4, spadRows = 256, accRows = 8
  )
  lazy val fp16MulFp32AddArithmeticImpl = new FPArithmeticImpl(5, 10, 8, 23)
  lazy val bf16MulFp32AddArithmeticImpl = new FPArithmeticImpl(8, 7, 8, 23)
  lazy val fp32ArithmeticImpl = new FPArithmeticImpl(8, 23, 8, 23)
  lazy val fp16ArithmeticImpl = new FPArithmeticImpl(5, 10, 5, 10)
  lazy val bf16ArithmeticImpl = new FPArithmeticImpl(8, 7, 8, 7)
}