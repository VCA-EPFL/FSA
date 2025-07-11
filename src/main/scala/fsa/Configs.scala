package fsa

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
import fsa.arithmetic._

case class FSAInjector[E <: Data : Arithmetic, A <: Data : Arithmetic](arithmeticImpl: ArithmeticImpl[E, A])
  extends SubsystemInjector((p, baseSubsystem) => {
  implicit val q = p
  val fsaParams = p(FSA)
  fsaParams.map{ params =>
    val fbus = baseSubsystem.locateTLBusWrapper(FBUS)
    val mbus = baseSubsystem.locateTLBusWrapper(MBUS)
    val fsa_domain = mbus.generateSynchronousDomain("fsa")
    val (fsa, tlConfigNode) = fsa_domain {
      val fsa = LazyModule(new AXI4FSA(arithmeticImpl))
      val tlConfigNode = TLEphemeralNode()
      // AXI4Deinterleaver is not needed since fsa never generate interleaved resp
      fsa.configNode :=
        AXI4UserYanker() :=
        TLToAXI4() :=
        TLFragmenter(fsa.instBeatBytes, fbus.blockBytes, holdFirstDeny = true) :=
        TLWidthWidget(fbus.beatBytes) :=
        tlConfigNode
      (fsa, tlConfigNode)
    }
    mbus.coupleFrom("fsa") {
      _ :=*
        AXI4ToTL() :=*
        AXI4UserYanker(capMaxFlight = Some(params.dmaMaxInflight)) :=*
        AXI4Fragmenter() :=*
        fsa.memNode
    }
    fbus.coupleTo("fsa") {
      mbus.crossIn(tlConfigNode)(ValName("fsa_fbus_xing"))(AsynchronousCrossing()) := _
    }
  }
})

class WithFpFSA
(
  params: FSAParams = Configs.fsa4x4,
  arithmeticImpl: ArithmeticImpl[FloatPoint, FloatPoint] = Configs.fp16MulFp32AddArithmeticImpl
) extends Config((site, here, up) => {
  case FSA => Some(params)
  case FpFSAImplKey => Some(arithmeticImpl)
})

class WithFpFSAMBusInjector extends Config((site, here, up) => {
  case SubsystemInjectorKey => up(SubsystemInjectorKey) + FSAInjector(site(FpFSAImplKey).get)
})

case object AXI4DirectMemPortKey extends Field[Option[MemoryPortParams]](None)

trait CanHaveFSADirectAXI4 { this: BaseSubsystem =>
  val fsaParams = p(FSA)
  val (fsaDomain, fsa, fsa_axi4) = fsaParams.map{ params =>
    val fbus = locateTLBusWrapper(FBUS)
    val mbus = locateTLBusWrapper(MBUS)
    val fsa_domain = mbus.generateSynchronousDomain("fsa")
    val (fsa, tlConfigNode) = fsa_domain {
      val fsa = LazyModule(new AXI4FSA(p(FpFSAImplKey).get))
      val tlConfigNode = TLEphemeralNode()
      // AXI4Deinterleaver is not needed since fsa never generate interleaved resp
      fsa.configNode :=
        AXI4UserYanker() :=
        TLToAXI4() :=
        TLFragmenter(fsa.instBeatBytes, fbus.blockBytes, holdFirstDeny = true) :=
        TLWidthWidget(fbus.beatBytes) :=
        tlConfigNode
      (fsa, tlConfigNode)
    }
    fbus.coupleTo("fsa") {
      mbus.crossIn(tlConfigNode)(ValName("fsa_fbus_xing"))(AsynchronousCrossing()) := _
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

    axi4SlaveNode :=* fsa.memNode
    val fsa_axi4 = InModuleBody{ axi4SlaveNode.makeIOs() }
    (fsa_domain, fsa, fsa_axi4)
  }.unzip3
}

object Configs {

  def defaultFSAParams(rows: Int, cols: Int, memPorts: Int): FSAParams = {
    /*
      SPAD:
      2 tile for Q in spad (cols)
      2x2 tiles for K and V for double buffering in spad
      Accumulator:
      1 row in accumulator for log exp sum
      1 tile for output O
    */
    FSAParams(
      rows, cols,
      // 2 tiles for Q, 2x2 tiles for K and V
      spadRows = 2 * cols + 4 * rows,
      // 1 row for log exp sum, 1 tile for output O
      accRows = 1 + rows,
      nMemPorts = memPorts,
    )
  }

  lazy val fsa4x4 = defaultFSAParams(4, 4, 4)
  lazy val fsa8x8 = defaultFSAParams(8, 8, 4)
  lazy val fsa16x16 = defaultFSAParams(16, 16, 8)
  lazy val fsa32x32 = defaultFSAParams(32, 32, 8)
  lazy val fsa64x64 = defaultFSAParams(64, 64, 8)
  lazy val fsa128x128 = defaultFSAParams(128, 128, 16)

  lazy val fp16MulFp32AddArithmeticImpl = new FPArithmeticImpl(5, 10, 8, 23)
  lazy val bf16MulFp32AddArithmeticImpl = new FPArithmeticImpl(8, 7, 8, 23)
  lazy val fp32ArithmeticImpl = new FPArithmeticImpl(8, 23, 8, 23)
  lazy val fp16ArithmeticImpl = new FPArithmeticImpl(5, 10, 5, 10)
  lazy val bf16ArithmeticImpl = new FPArithmeticImpl(8, 7, 8, 7)
}