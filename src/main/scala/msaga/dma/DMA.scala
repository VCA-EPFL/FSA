package msaga.dma

import org.chipsalliance.diplomacy.lazymodule._
import freechips.rocketchip.amba.axi4._
import freechips.rocketchip.diplomacy._
import org.chipsalliance.cde.config._
import chisel3._
import chisel3.util._
import msaga.frontend.Semaphore
import msaga.{SRAMRead, SRAMWrite}
import msaga.isa.DMAInstruction
import msaga.isa.ISA.DMAFunc
import msaga.utils.{DelayedAssert, Ehr}


class DMAImpl[E <: Data, A <: Data](outer: DMA[E, A]) extends LazyModuleImp(outer) {
  val node = outer.node
  val nPorts = node.out.size
  val memAddrWidth = node.out.map(_._2.bundle.addrBits).max
  val sramAddrWidth = outer.sramAddrWidth
  val (dmaLoadInflight, dmaStoreInflight) = (outer.dmaLoadInflight, outer.dmaStoreInflight)

  val io = IO(new Bundle {
    val inst = Flipped(Decoupled(new DMAInstruction(sramAddrWidth, memAddrWidth)))
    val semaphoreAcquire = Decoupled(new Semaphore)
    val semaphoreRelease = Valid(new Semaphore)
    val spadWrite = Vec(nPorts, new SRAMWrite(sramAddrWidth, outer.spadElem, outer.spadCols))
    val accRead = Vec(nPorts, new SRAMRead(sramAddrWidth, outer.accElem, outer.accCols))
    val busy = Output(Bool())
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

  val acqFlag = Ehr(2, Bool(), Some(false.B))
  val outReq = partitioner.io.out.bits.head
  io.semaphoreAcquire.valid := partitioner.io.out.valid && outReq.acquireValid
  io.semaphoreAcquire.bits.id := outReq.semId
  io.semaphoreAcquire.bits.value := outReq.acquireSemValue
  when(io.semaphoreAcquire.fire) {
    acqFlag.write(0, true.B)
  }
  when(partitioner.io.out.fire) {
    acqFlag.write(1, false.B)
  }
  val depReady = !outReq.acquireValid || acqFlag.read(1)
  val (loadQueues, storeQueues) = io.spadWrite.zip(io.accRead).zip(partitioner.io.out.bits).zip(node.out).map{
    case (((spad, acc), req), (axi, edge)) =>
      val loadHandler = Module(new LoadQueue(edge, chiselTypeOf(dmaReq.bits), dmaLoadInflight, spad))
      val storeHandler = Module(new StoreQueue(edge, chiselTypeOf(dmaReq.bits), dmaStoreInflight, acc))
      loadHandler.io.req.valid := partitioner.io.out.valid && outReq.isLoad && depReady
      loadHandler.io.req.bits := req
      storeHandler.io.req.valid := partitioner.io.out.valid && !outReq.isLoad && depReady
      storeHandler.io.req.bits := req
      axi.ar <> loadHandler.ar
      axi.aw <> storeHandler.aw
      loadHandler.r <> axi.r
      axi.w <> storeHandler.w
      storeHandler.b <> axi.b
      spad <> loadHandler.spadWrite
      acc <> storeHandler.accRead
      (loadHandler, storeHandler)
  }.unzip

  val loadReady = Cat(loadQueues.map(_.io.req.ready)).andR
  val storeReady = Cat(storeQueues.map(_.io.req.ready)).andR
  partitioner.io.out.ready := Mux(outReq.isLoad, loadReady, storeReady) && depReady


  io.busy := Cat(
    dmaReq.valid,
    partitioner.io.out.valid,
    Cat(loadQueues.map(_.io.busy)).orR,
    Cat(storeQueues.map(_.io.busy)).orR
  ).orR

  val loadSemWrite = Wire(Decoupled(new Semaphore))
  val storeSemWrite = Wire(Decoupled(new Semaphore))

  loadSemWrite.valid := Cat(loadQueues.map(_.io.semRelease.valid)).andR
  loadSemWrite.bits := loadQueues.head.io.semRelease.bits
  loadQueues.foreach(_.io.semRelease.ready := loadSemWrite.fire)

  storeSemWrite.valid := Cat(storeQueues.map(_.io.semRelease.valid)).andR
  storeSemWrite.bits := storeQueues.head.io.semRelease.bits
  storeQueues.foreach(_.io.semRelease.ready := storeSemWrite.fire)

  val arb = Module(new Arbiter(new Semaphore, 2))
  arb.io.in(0) <> loadSemWrite
  arb.io.in(1) <> storeSemWrite
  arb.io.out.ready := true.B
  io.semaphoreRelease.valid := arb.io.out.valid && Mux(arb.io.in(0).fire,
    loadQueues.head.io.doSemRelease,
    storeQueues.head.io.doSemRelease
  )
  io.semaphoreRelease.bits := arb.io.out.bits

  for (i <- 1 until nPorts) {
    DelayedAssert(!loadSemWrite.fire ||
      loadQueues(i).io.semRelease.bits.asUInt === io.semaphoreRelease.bits.asUInt
    )
  }
  for (i <- 1 until nPorts) {
    DelayedAssert(!storeSemWrite.fire ||
        storeQueues(i).io.semRelease.bits.asUInt === io.semaphoreRelease.bits.asUInt
    )
  }

}

class DMA[E <: Data, A <: Data]
(
  val nPorts: Int,
  val sramAddrWidth: Int,
  val dmaLoadInflight: Int,
  val dmaStoreInflight: Int,
  val spadElem: E,
  val spadCols: Int,
  val accElem: A,
  val accCols: Int
)(implicit p: Parameters) extends LazyModule {

  require(isPow2(nPorts))

  val node = AXI4MasterNode(Seq.fill(nPorts){AXI4MasterPortParameters(
    masters = Seq(AXI4MasterParameters(
      name = "dma", id = IdRange(0, 1)
    ))
  )})

  lazy val module = new DMAImpl(this)
}