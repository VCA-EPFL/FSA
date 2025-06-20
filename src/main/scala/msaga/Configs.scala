package msaga

import org.chipsalliance.cde.config._
import org.chipsalliance.diplomacy.lazymodule.LazyModule
import org.chipsalliance.diplomacy.ValName
import testchipip.soc.{SubsystemInjector, SubsystemInjectorKey}
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.subsystem._
import freechips.rocketchip.amba.axi4._
import freechips.rocketchip.tilelink._
import freechips.rocketchip.prci.AsynchronousCrossing
import chisel3._
import msaga.arithmetic._

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
      mbus.crossIn(tlConfigNode)(ValName("msaga_fbus_xing"))(AsynchronousCrossing()) := _
    }
  }
})

class WithFpMSAGA
(
  params: MSAGAParams = Configs.msaga4x4,
  arithmeticImpl: ArithmeticImpl[FloatPoint, FloatPoint] = Configs.fp16MulFp32AddArithmeticImpl
) extends Config((site, here, up) => {
  case MSAGAKey => Some(params)
  case FpMSAGAImplKey => Some(arithmeticImpl)
})

class WithFpMSAGAMBusInjector extends Config((site, here, up) => {
  case SubsystemInjectorKey => up(SubsystemInjectorKey) + MSAGAInjector(site(FpMSAGAImplKey).get)
})

case object AXI4DirectMemPortKey extends Field[Option[MemoryPortParams]](None)

trait CanHaveMSAGADirectAXI4 { this: BaseSubsystem =>
  val msagaParams = p(MSAGAKey)
  val (msagaDomain, msaga, msaga_axi4) = msagaParams.map{ params =>
    val fbus = locateTLBusWrapper(FBUS)
    val mbus = locateTLBusWrapper(MBUS)
    val domain = mbus.generateSynchronousDomain("msaga")
    val (msaga, tlConfigNode) = domain {
      val msaga = LazyModule(new AXI4MSAGA(p(FpMSAGAImplKey).get))
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
    fbus.coupleTo("msaga") {
      mbus.crossIn(tlConfigNode)(ValName("msaga_fbus_xing"))(AsynchronousCrossing()) := _
    }

    val memPortParamsOpt = p(AXI4DirectMemPortKey)
    val axi4SlaveNode = AXI4SlaveNode(memPortParamsOpt.map({ case MemoryPortParams(memPortParams, nMemoryChannels, _) =>
      Seq.tabulate(nMemoryChannels) { channel =>
        val base = AddressSet.misaligned(memPortParams.base, memPortParams.size)
        val blockBytes = memPortParams.maxXferBytes
        val filter = AddressSet(channel * blockBytes, ~((nMemoryChannels-1) * blockBytes))

        AXI4SlavePortParameters(
          slaves = Seq(AXI4SlaveParameters(
            address       = base.flatMap(_.intersect(filter)),
            regionType    = RegionType.UNCACHED, // cacheable
            executable    = true,
            supportsWrite = TransferSizes(1, blockBytes),
            supportsRead  = TransferSizes(1, blockBytes),
            interleavedId = Some(0))), // slave does not interleave read responses
          beatBytes = memPortParams.beatBytes)
      }
    }).toList.flatten)

    axi4SlaveNode :=* msaga.memNode
    val msaga_axi4 = InModuleBody{ axi4SlaveNode.makeIOs() }
    (domain, msaga, msaga_axi4)
  }.unzip3
}

object Configs {

  def defaultMSAGAParams(rows: Int, cols: Int, memPorts: Int): MSAGAParams = {
    /*
      SPAD:
      2 tile for Q in spad (cols)
      2x2 tiles for K and V for double buffering in spad
      Accumulator:
      1 row in accumulator for log exp sum
      1 tile for output O
    */
    MSAGAParams(
      rows, cols,
      // 2 tiles for Q, 2x2 tiles for K and V
      spadRows = 2 * cols + 4 * rows,
      // 1 row for log exp sum, 1 tile for output O
      accRows = 1 + rows,
      nMemPorts = memPorts,
    )
  }

  lazy val msaga4x4 = defaultMSAGAParams(4, 4, 1)
  lazy val msaga8x8 = defaultMSAGAParams(8, 8, 1)
  lazy val msaga16x16 = defaultMSAGAParams(16, 16, 2)
  lazy val msaga32x32 = defaultMSAGAParams(32, 32, 4)
  lazy val msaga64x64 = defaultMSAGAParams(64, 64, 8)
  lazy val msaga128x128 = defaultMSAGAParams(128, 128, 16)

  lazy val fp16MulFp32AddArithmeticImpl = new FPArithmeticImpl(5, 10, 8, 23)
  lazy val bf16MulFp32AddArithmeticImpl = new FPArithmeticImpl(8, 7, 8, 23)
  lazy val fp32ArithmeticImpl = new FPArithmeticImpl(8, 23, 8, 23)
  lazy val fp16ArithmeticImpl = new FPArithmeticImpl(5, 10, 5, 10)
  lazy val bf16ArithmeticImpl = new FPArithmeticImpl(8, 7, 8, 7)
}