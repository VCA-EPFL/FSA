package fsa.dma

import chisel3._
import chisel3.util._
import freechips.rocketchip.amba.axi4._
import fsa.{SRAMNarrowRead, SRAMNarrowWrite}
import fsa.frontend.Semaphore
import fsa.isa.ISA.Constants._
import fsa.utils.{DelayedAssert, Ehr}

abstract class BaseLoadStoreQueue(reqGen: DMARequest, n: Int) extends Module {

  val io = IO(new Bundle {
    val req = Flipped(Decoupled(reqGen))
    val semAcquire = Decoupled(new Semaphore)
    val semRelease = Decoupled(new Semaphore)
    val doSemRelease = Output(Bool())
    val busy = Output(Bool())
    val active = Output(Bool())
  })

  class Entry extends Bundle {
    val req = reqGen.cloneType
    val rRepeat = UInt(DMA_REPEAT_BITS.W)
    val depReady = Bool()
  }

  val entryValid = RegInit(VecInit(Seq.fill(n)(false.B)))
  val entries = Reg(Vec(n, new Entry))

  val enqPtr = Counter(n)
  val acqPtr = Counter(n)
  val deqPtr = Counter(n)

  // acquire semaphore before executing the request
  val acqEntry = entries(acqPtr.value)
  io.semAcquire.valid := entryValid(acqPtr.value) && acqEntry.req.acquireValid
  io.semAcquire.bits.id := acqEntry.req.semId
  io.semAcquire.bits.value := acqEntry.req.acquireSemValue
  when(io.semAcquire.fire || entryValid(acqPtr.value) && !acqEntry.req.acquireValid) {
    acqEntry.req.acquireValid := false.B
    acqEntry.depReady := true.B
    acqPtr.inc()
  }

  def valid(ptr: Counter): Bool = {
    entryValid(ptr.value) && entries(ptr.value).depReady
  }

  // deq request
  val deqEntry = entries(deqPtr.value)
  io.semRelease.valid := valid(deqPtr) && deqEntry.rRepeat === 0.U
  io.semRelease.bits.id := deqEntry.req.semId
  io.semRelease.bits.value := deqEntry.req.releaseSemValue
  io.doSemRelease := deqEntry.req.releaseValid
  when(io.semRelease.fire) {
    entryValid(deqPtr.value) := false.B
    deqPtr.inc()
  }

  // enq request
  when(io.req.fire) {
    entries(enqPtr.value).req := io.req.bits
    entries(enqPtr.value).rRepeat := io.req.bits.repeat
    entries(enqPtr.value).depReady := false.B
    entryValid(enqPtr.value) := true.B
    enqPtr.inc()
  }

  io.req.ready := !entryValid(enqPtr.value)
  io.busy := entryValid(deqPtr.value)
  io.active := valid(deqPtr)
}


class StoreQueue
(
  edge: AXI4EdgeParameters,
  reqGen: DMARequest, nInflight: Int,
  accReadGen: SRAMNarrowRead
) extends BaseLoadStoreQueue(reqGen, nInflight) {

  val aw = IO(Decoupled(new AXI4BundleAW(edge.bundle)))
  val w = IO(Decoupled(new AXI4BundleW(edge.bundle)))
  val b = IO(Flipped(Decoupled(new AXI4BundleB(edge.bundle))))
  val accRead = IO(accReadGen.cloneType)

  val awPtr = Counter(nInflight)
  val rPtr = Counter(nInflight)

  val awEntry = entries(awPtr.value)

  aw.valid := valid(awPtr) && awEntry.req.repeat =/= 0.U
  aw.bits.addr := awEntry.req.memAddr
  aw.bits.addr := awEntry.req.memAddr
  aw.bits.id := 0.U
  aw.bits.len := (awEntry.req.size >> log2Up(edge.slave.beatBytes)).asUInt - 1.U
  aw.bits.size := log2Up(edge.slave.beatBytes).U
  aw.bits.burst := AXI4Parameters.BURST_INCR
  aw.bits.lock := 0.U
  aw.bits.cache := 0.U
  aw.bits.prot := 0.U
  aw.bits.qos := 0.U

  when(aw.fire) {
    awEntry.req.repeat := awEntry.req.repeat - 1.U
    awEntry.req.memAddr := (awEntry.req.memAddr.asSInt + awEntry.req.memStride).asUInt
    when(awEntry.req.repeat === 1.U) {
      awPtr.inc()
    }
  }

  val beatBits = edge.slave.beatBytes * 8
  val nBeats = accRead.nSubBanks

  // sram read
  val rBeatCnt = RegInit(0.U(log2Up(nBeats).W))
  val rLast = rBeatCnt === (nBeats - 1).U
  val writeQueue = Module(new Queue(UInt(beatBits.W), entries = 2, pipe = true))
  DelayedAssert(!writeQueue.io.enq.valid || writeQueue.io.enq.ready)
  val queueCntNext = Mux(writeQueue.io.enq.valid,
    Mux(writeQueue.io.deq.fire, writeQueue.io.count, writeQueue.io.count + 1.U),
    Mux(writeQueue.io.deq.fire, writeQueue.io.count - 1.U, writeQueue.io.count)
  )
  val rEntry = entries(rPtr.value)
  accRead.valid := valid(rPtr) && queueCntNext < writeQueue.entries.U
  accRead.addr := rEntry.req.sramAddr
  accRead.subBankIdx := rBeatCnt
  writeQueue.io.enq.valid := RegNext(accRead.fire, init = false.B)
  writeQueue.io.enq.bits := accRead.data.asUInt
  when(accRead.fire) {
    rBeatCnt := Mux(rLast, 0.U, rBeatCnt + 1.U)
    when(rLast) {
      rEntry.rRepeat := rEntry.rRepeat - 1.U
      rEntry.req.sramAddr := (rEntry.req.sramAddr.asSInt + rEntry.req.sramStride).asUInt
      when(rEntry.rRepeat === 1.U) {
        rPtr.inc()
      }
    }
  }

  // mem write
  val wBeatCnt = RegInit(0.U(log2Up(nBeats).W))
  w.valid := writeQueue.io.deq.valid
  w.bits.data := writeQueue.io.deq.bits
  w.bits.strb := ~0.U(w.bits.strb.getWidth.W)
  w.bits.last := wBeatCnt === (nBeats - 1).U
  writeQueue.io.deq.ready := w.ready
  b.ready := true.B
  assert(b.bits.id === 0.U && b.bits.resp === AXI4Parameters.RESP_OKAY)
  when(w.fire) {
    wBeatCnt := Mux(w.bits.last, 0.U, wBeatCnt + 1.U)
  }

}


class LoadQueue[E <: Data]
(
  edge: AXI4EdgeParameters, reqGen: DMARequest,
  nInflight: Int, spadWriteGen: SRAMNarrowWrite
) extends BaseLoadStoreQueue(reqGen, nInflight) {

  val ar = IO(Decoupled(new AXI4BundleAR(edge.bundle)))
  val r = IO(Flipped(Decoupled(new AXI4BundleR(edge.bundle))))
  val spadWrite = IO(spadWriteGen.cloneType)

  val arPtr = Counter(nInflight)
  val rPtr = Counter(nInflight)

  // read address
  val arEntry = entries(arPtr.value)
  ar.valid := valid(arPtr) && arEntry.req.repeat =/= 0.U
  ar.bits.addr := arEntry.req.memAddr
  ar.bits.id := 0.U
  ar.bits.len := (arEntry.req.size >> log2Up(edge.slave.beatBytes)).asUInt - 1.U
  ar.bits.size := log2Up(edge.slave.beatBytes).U
  ar.bits.burst := AXI4Parameters.BURST_INCR
  ar.bits.lock := 0.U
  ar.bits.cache := 0.U
  ar.bits.prot := 0.U
  ar.bits.qos := 0.U

  when(ar.fire) {
    arEntry.req.repeat := arEntry.req.repeat - 1.U
    arEntry.req.memAddr := (arEntry.req.memAddr.asSInt + arEntry.req.memStride).asUInt
    when(arEntry.req.repeat === 1.U) {
      arPtr.inc()
    }
  }

  // read data
  val nBeats = spadWrite.nSubBanks
  val rBeatCnt = RegInit(0.U(log2Up(nBeats).W))
  spadWrite.valid := r.valid
  spadWrite.addr := entries(rPtr.value).req.sramAddr
  spadWrite.data := r.bits.data.asTypeOf(spadWrite.data)
  spadWrite.subBankIdx := rBeatCnt
  r.ready := spadWrite.ready

  when(r.fire) {
    val rEntry = entries(rPtr.value)
    assert(r.bits.id === 0.U, "Currently only one ID is supported")
    assert(r.bits.resp === AXI4Parameters.RESP_OKAY, "Currently only OKAY response is supported")
    rBeatCnt := Mux(r.bits.last, 0.U, rBeatCnt + 1.U)
    when(r.bits.last) {
      rEntry.rRepeat := rEntry.rRepeat - 1.U
      rEntry.req.sramAddr := (rEntry.req.sramAddr.asSInt + rEntry.req.sramStride).asUInt
      when(rEntry.rRepeat === 1.U) {
        rPtr.inc()
      }
    }
  }
}
