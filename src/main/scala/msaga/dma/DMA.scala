package msaga.dma

import org.chipsalliance.diplomacy.lazymodule._
import freechips.rocketchip.amba.axi4._
import freechips.rocketchip.diplomacy._
import org.chipsalliance.cde.config._
import chisel3._
import chisel3.util._
import msaga.frontend.Semaphore
import msaga.{SRAMNarrowRead, SRAMNarrowWrite}
import msaga.isa.DMAInstruction
import msaga.isa.ISA.DMAFunc
import msaga.utils.{DelayedAssert, Ehr}


class DMAImpl(outer: DMA) extends LazyModuleImp(outer) {
  val node = outer.node
  val nPorts = node.out.size
  val memAddrWidth = node.out.map(_._2.bundle.addrBits).max
  val sramAddrWidth = outer.sramAddrWidth
  val (dmaLoadInflight, dmaStoreInflight) = (outer.dmaLoadInflight, outer.dmaStoreInflight)

  val beatBytes = node.out.head._2.slave.beatBytes

  val io = IO(new Bundle {
    val inst = Flipped(Decoupled(new DMAInstruction(sramAddrWidth, memAddrWidth)))
    // one for load and one for storek
    val semaphoreAcquire = Vec(2, Decoupled(new Semaphore))
    val semaphoreRelease = Vec(2, Valid(new Semaphore))
    val spadWrite = Vec(nPorts, new SRAMNarrowWrite(sramAddrWidth, outer.spadElemWidth, outer.spadRowSize, beatBytes))
    val accRead = Vec(nPorts, new SRAMNarrowRead(sramAddrWidth, outer.accElemWidth, outer.accRowSize, beatBytes))
    val busy = Output(Bool())
    val active = Output(Bool())
  })

  val dmaReq = Wire(Decoupled(new DMARequest(sramAddrWidth, memAddrWidth)))
  dmaReq.valid := io.inst.valid
  dmaReq.bits.memAddr := io.inst.bits.mem.addr
  dmaReq.bits.memStride := io.inst.bits.mem.stride
  dmaReq.bits.sramAddr := io.inst.bits.sram.addr
  dmaReq.bits.sramStride := io.inst.bits.sram.stride
  dmaReq.bits.repeat := io.inst.bits.header.repeat
  dmaReq.bits.size := io.inst.bits.mem.size
  dmaReq.bits.semId := io.inst.bits.header.semId
  dmaReq.bits.acquireValid := io.inst.bits.header.acquireValid
  dmaReq.bits.acquireSemValue := io.inst.bits.header.acquireSemValue
  dmaReq.bits.releaseValid := io.inst.bits.header.releaseValid
  dmaReq.bits.releaseSemValue := io.inst.bits.header.releaseSemValue
  dmaReq.bits.isLoad := io.inst.bits.header.func === DMAFunc.LD_SRAM
  io.inst.ready := dmaReq.ready

  val partitioner = Module(new RequestPartitioner(chiselTypeOf(dmaReq.bits), nPorts))
  partitioner.io.in <> dmaReq

  val outReq = partitioner.io.out.bits.head

  val (loadQueues, storeQueues) = io.spadWrite.zip(io.accRead).zip(partitioner.io.out.bits).zip(node.out).map{
    case (((spad, acc), req), (axi, edge)) =>
      val loadQueue = Module(new LoadQueue(edge, chiselTypeOf(dmaReq.bits), dmaLoadInflight, spad))
      val storeQueue = Module(new StoreQueue(edge, chiselTypeOf(dmaReq.bits), dmaStoreInflight, acc))
      loadQueue.io.req.valid := partitioner.io.out.valid && outReq.isLoad
      loadQueue.io.req.bits := req
      storeQueue.io.req.valid := partitioner.io.out.valid && !outReq.isLoad
      storeQueue.io.req.bits := req
      axi.ar <> loadQueue.ar
      axi.aw <> storeQueue.aw
      loadQueue.r <> axi.r
      axi.w <> storeQueue.w
      storeQueue.b <> axi.b
      spad <> loadQueue.spadWrite
      acc <> storeQueue.accRead
      (loadQueue, storeQueue)
  }.unzip

  /* To simplify the design, we deq all load queues or all store queues
     at the same time, so their enq ready signals should be synchronized.
   */
  partitioner.io.out.ready := Mux(outReq.isLoad,
    loadQueues.head.io.req.ready,
    storeQueues.head.io.req.ready
  )

  io.active := loadQueues.head.io.active || storeQueues.head.io.active
  io.busy := loadQueues.head.io.busy || storeQueues.head.io.busy || dmaReq.valid || partitioner.io.out.valid

  val loadSemRelease = Wire(Valid(new Semaphore))
  val storeSemRelease = Wire(Valid(new Semaphore))

  // wait until all load finish
  loadSemRelease.valid := Cat(loadQueues.map(_.io.semRelease.valid)).andR && loadQueues.head.io.doSemRelease
  loadSemRelease.bits := loadQueues.head.io.semRelease.bits
  loadQueues.foreach(_.io.semRelease.ready := loadSemRelease.fire)

  // wait until all store finish
  storeSemRelease.valid := Cat(storeQueues.map(_.io.semRelease.valid)).andR && storeQueues.head.io.doSemRelease
  storeSemRelease.bits := storeQueues.head.io.semRelease.bits
  storeQueues.foreach(_.io.semRelease.ready := storeSemRelease.fire)

  io.semaphoreRelease.head <> loadSemRelease
  io.semaphoreRelease.last <> storeSemRelease

  io.semaphoreAcquire.head <> loadQueues.head.io.semAcquire
  io.semaphoreAcquire.last <> storeQueues.head.io.semAcquire
  loadQueues.tail.foreach(_.io.semAcquire.ready := io.semaphoreAcquire.head.ready)
  storeQueues.tail.foreach(_.io.semAcquire.ready := io.semaphoreAcquire.last.ready)

}

class DMA
(
  val nPorts: Int,
  val sramAddrWidth: Int,
  val dmaLoadInflight: Int,
  val dmaStoreInflight: Int,
  val spadElemWidth: Int,
  val spadRowSize: Int,
  val accElemWidth: Int,
  val accRowSize: Int
)(implicit p: Parameters) extends LazyModule {

  require(isPow2(nPorts))

  val node = AXI4MasterNode(Seq.fill(nPorts){AXI4MasterPortParameters(
    masters = Seq(AXI4MasterParameters(
      name = "dma", id = IdRange(0, 1)
    ))
  )})

  lazy val module = new DMAImpl(this)
}