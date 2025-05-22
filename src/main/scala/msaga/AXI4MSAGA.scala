package msaga

import chisel3._
import chisel3.util._
import org.chipsalliance.diplomacy.lazymodule._
import org.chipsalliance.cde.config._
import freechips.rocketchip.amba.axi4._
import freechips.rocketchip.diplomacy.AddressSet
import freechips.rocketchip.regmapper.RegField
import msaga.frontend.{Decoder, Semaphores}
import msaga.arithmetic._
import msaga.dma.DMA
import msaga.utils.Ehr

class AXI4MSAGA[E <: Data : Arithmetic, A <: Data : Arithmetic](val ev: ArithmeticImpl[E, A])(implicit p: Parameters) extends LazyModule {
  val instBeatBytes = 4
  val instBeatBits = instBeatBytes * 8
  val msagaParams = p(MSAGAKey).get

  val configNode = AXI4RegisterNode(
    address = AddressSet(0x8000, 0xff)
  )
  val dma = LazyModule(new DMA(
    nPorts = msagaParams.nMemPorts,
    sramAddrWidth = msagaParams.sramAddrWidth,
    dmaLoadInflight = msagaParams.dmaLoadInflight,
    dmaStoreInflight = msagaParams.dmaStoreInflight,
    spadElem = ev.elemType,
    spadCols = msagaParams.saRows,
    accElem = ev.accType,
    accCols = msagaParams.saCols
  ))
  val memNode = dma.node

  lazy val module = new LazyModuleImp(this) {

    val memAddrWidth = memNode.out.map(_._2.bundle.addrBits).max

    val s_idle :: s_active :: s_done :: Nil = Enum(3)
    val state = RegInit(s_idle)
    val set_active = WireInit(false.B)
    val set_done = Wire(Bool())
    val rawInstQueue = Module(
      new Queue(UInt(instBeatBits.W), msagaParams.instructionQueueEntries, useSyncReadMem = true)
    )
    configNode.regmap(
      0x00 -> Seq(RegField.w(instBeatBits, rawInstQueue.io.enq)),
      0x04 -> Seq(RegField.w(1, set_active)),
      0x08 -> Seq(RegField.r(2, state))
    )

    switch(state) {
      is(s_idle) {
        when(set_active) {
          state := s_active
        }
      }
      is(s_active) {
        when(set_done) {
          state := s_done
        }
      }
      is(s_done) {
        when(set_active) {
          state := s_active
        }
      }
    }


    val decoder = Module(new Decoder(memAddrWidth))
    val semaphores = Module(new Semaphores(nRead = 2, nWrite = 2))
    val msaga = Module(new MSAGA(ev))

    val is_active = state === s_active
    decoder.io.in.valid := rawInstQueue.io.deq.valid && is_active
    decoder.io.in.bits := rawInstQueue.io.deq.bits
    rawInstQueue.io.deq.ready := decoder.io.in.ready && is_active

    val mxInst = Queue(decoder.io.outMx, entries = msagaParams.mxInflight, pipe = true)
    // DMA has its own load/store queues inside it
    val dmaInst = Queue(decoder.io.outDMA, pipe = true)

    val dmaDone = RegNext(!(dma.module.io.busy || dmaInst.valid), init = false.B)
    val mxDone = RegNext(!(msaga.io.busy || mxInst.valid), init = false.B)
    val fenceReady = (!decoder.io.outFence.bits.dma || dmaDone) &&
      (!decoder.io.outFence.bits.dma || mxDone)
    decoder.io.outFence.ready := fenceReady
    set_done := decoder.io.outFence.fire && decoder.io.outFence.bits.stop


    val mxSemAcquire = semaphores.io.acquire.head
    val mxAcqFlag = Ehr(2, Bool(), Some(false.B))
    val mxSemRelease = semaphores.io.release.head
    mxSemAcquire.valid := mxInst.bits.header.acquireValid && mxInst.valid
    mxSemAcquire.bits.id := mxInst.bits.header.semId
    mxSemAcquire.bits.value := mxInst.bits.header.acquireSemValue
    when(mxSemAcquire.fire){
      mxAcqFlag.write(0, true.B)
    }
    when(msaga.io.inst.fire) {
      mxAcqFlag.write(1, false.B)
    }
    val mxDepReady = !mxInst.bits.header.acquireValid || mxAcqFlag.read(1)

    msaga.io.inst.valid := mxInst.valid && mxDepReady
    msaga.io.inst.bits := mxInst.bits
    mxInst.ready := msaga.io.inst.ready && mxDepReady

    mxSemRelease <> msaga.io.sem_release

    msaga.io.spad_write <> dma.module.io.spadWrite
    msaga.io.acc_read <> dma.module.io.accRead

    dma.module.io.inst <> dmaInst

    val dmaSemAcquire = semaphores.io.acquire.last
    val dmaSemRelease = semaphores.io.release.last

    dmaSemAcquire <> dma.module.io.semaphoreAcquire
    dmaSemRelease <> dma.module.io.semaphoreRelease

  }
}

