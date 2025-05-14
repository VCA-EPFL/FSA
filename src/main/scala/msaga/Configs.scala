package msaga

import org.chipsalliance.diplomacy.lazymodule._
import org.chipsalliance.cde.config._
import org.chipsalliance.diplomacy.ValName
import testchipip.soc.{SubsystemInjector, SubsystemInjectorKey}
import freechips.rocketchip.subsystem._
import freechips.rocketchip.amba.axi4._
import freechips.rocketchip.tilelink._
import freechips.rocketchip.prci.SynchronousCrossing
import msaga.arithmetic._
import chisel3._
import chisel3.util._
import msaga.dma.DMA

case class MSAGAInjector[E <: Data : Arithmetic, A <: Data : Arithmetic](arithmeticImpl: ArithmeticImpl[E, A])
  extends SubsystemInjector((p, baseSubsystem) => {
  implicit val q = p
  val msagaParams = p(MSAGAKey)
  msagaParams.map{ params =>
    val fbus = baseSubsystem.locateTLBusWrapper(FBUS)
    val mbus = baseSubsystem.locateTLBusWrapper(MBUS)
    val domain = mbus.generateSynchronousDomain("msaga")
    val (msaga, tlConfigNode) = domain {
      val msaga = LazyModule(new AXI4MSAGA(arithmeticImpl))
      val tlConfigNode = TLEphemeralNode()
      // AXI4Deinterleaver is not needed since msaga never generate interleaved resp
      msaga.configNode :=
        AXI4UserYanker() :=
        TLToAXI4() :=
        TLFragmenter(msaga.instBeatBytes, fbus.blockBytes, holdFirstDeny = true) :=
        TLWidthWidget(fbus.beatBytes) :=
        tlConfigNode
      (msaga, tlConfigNode)
    }
    mbus.coupleFrom("msaga") {
      _ :=*
        AXI4ToTL() :=*
        AXI4UserYanker(capMaxFlight = Some(params.dmaMaxInflight)) :=*
        AXI4Fragmenter() :=*
        msaga.memNode
    }
    fbus.coupleTo("msaga") {
      mbus.crossIn(tlConfigNode)(ValName("msaga_fbus_xing"))(SynchronousCrossing()) := _
    }
  }
})

case object DMAInjector extends SubsystemInjector((p, baseSubsystem) => {
  implicit val q = p
  val mbus = baseSubsystem.locateTLBusWrapper(MBUS)
  val dma = mbus {
    // LazyModule(new DMATester())
    LazyModule(new DMA(p(ExtMem).get.nMemoryChannels, 10, 16, 16))
  }
  mbus.coupleFrom("dma") {
    _ :=*
      AXI4ToTL() :=*
      AXI4UserYanker(capMaxFlight = Some(15)) :=*
      AXI4Fragmenter() :=*
      dma.node
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