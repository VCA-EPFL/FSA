package msaga

import chisel3._
import chisel3.util._
import org.chipsalliance.diplomacy.lazymodule._
import org.chipsalliance.cde.config._
import freechips.rocketchip.amba.axi4._
import freechips.rocketchip.diplomacy.AddressSet
import freechips.rocketchip.regmapper.RegField
import freechips.rocketchip.subsystem.ExtMem
import msaga.frontend.{Decoder, Semaphores}
import msaga.arithmetic._
import msaga.dma.DMA

class AXI4MSAGA[E <: Data : Arithmetic, A <: Data : Arithmetic](val ev: ArithmeticImpl[E, A])(implicit p: Parameters) extends LazyModule {
  val instBeatBytes = 4
  val instBeatBits = instBeatBytes * 8
  val msagaParams = p(MSAGAKey).get

  val configNode = AXI4RegisterNode(
    address = AddressSet(0x8000, 0xff)
  )
  val dma = LazyModule(new DMA(
    nPorts = msagaParams.nMemPorts.getOrElse(p(ExtMem).map(_.nMemoryChannels).getOrElse(1)),
    sramAddrWidth = msagaParams.sramAddrWidth,
    dmaLoadInflight = msagaParams.dmaLoadInflight,
    dmaStoreInflight = msagaParams.dmaStoreInflight,
  ))
  val memNode = dma.node

  lazy val module = new LazyModuleImp(this) {

    val memAddrWidth = memNode.out.map(_._2.bundle.addrBits).max

    val busy = RegInit(false.B)
    val rawInstQueue = Module(new Queue(UInt(instBeatBits.W), msagaParams.instructionQueueEntries, useSyncReadMem = true))
    val decoder = Module(new Decoder(memAddrWidth))
    val semaphores = Module(new Semaphores(nRead = 2, nWrite = 2))
    val msaga = Module(new MSAGA(ev))

    configNode.regmap(
      0x00 -> Seq(RegField.w(instBeatBits, rawInstQueue.io.enq)),
      0x04 -> Seq(RegField.w(1, busy))
    )

    decoder.io.in.valid := rawInstQueue.io.deq.valid && busy
    decoder.io.in.bits := rawInstQueue.io.deq.bits
    rawInstQueue.io.deq.ready := decoder.io.in.ready && busy

    val mxInst = Queue(decoder.io.outMx)
    val dmaInst = Queue(decoder.io.outDMA)

    semaphores.io.read.head.semaphoreId := mxInst.bits.header.consumerSemId
    semaphores.io.read.head.semaphoreValue := mxInst.bits.header.consumerSemValue
    semaphores.io.read.last.semaphoreId := dmaInst.bits.header.consumerSemId
    semaphores.io.read.last.semaphoreValue := dmaInst.bits.header.consumerSemValue

    val mxReady :: dmaReady :: Nil = semaphores.io.read.map(_.ready).toList

    msaga.io.inst.valid := mxInst.valid && mxReady
    msaga.io.inst.bits := mxInst.bits
    mxInst.ready := msaga.io.inst.ready && mxReady

    dma.module.io.inst <> dmaInst

    //TODO
    msaga.io.debug_sram_io <> DontCare
    semaphores.io.write.foreach{w =>
      w.valid := false.B
      w.bits := DontCare
    }
  }
}

