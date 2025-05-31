package msaga.dma

import chisel3._
import chisel3.util._
import freechips.rocketchip.amba.axi4._
import msaga.frontend.Semaphore
import msaga.{SRAMRead, SRAMWrite}
import msaga.isa.ISA.Constants._
import msaga.utils.DelayedAssert

abstract class BaseLoadStoreQueue(reqGen: DMARequest, n: Int) extends Module {

  val io = IO(new Bundle {
    val req = Flipped(Decoupled(reqGen))
    val semRelease = Decoupled(new Semaphore)
    val doSemRelease = Output(Bool())
    val busy = Output(Bool())
  })

  class Entry extends Bundle {
    val req = reqGen.cloneType
    val rRepeat = UInt(DMA_REPEAT_BITS.W)
  }

  val entryValid = RegInit(VecInit(Seq.fill(n)(false.B)))
  val entries = Reg(Vec(n, new Entry))

  val enqPtr = Counter(n)
  val deqPtr = Counter(n)

  // deq request
  val deqEntry = entries(deqPtr.value)
  io.semRelease.valid := entryValid(deqPtr.value) && deqEntry.rRepeat === 0.U
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
    entryValid(enqPtr.value) := true.B
    enqPtr.inc()
  }

  io.req.ready := !entryValid(enqPtr.value)
  io.busy := entryValid(deqPtr.value)
}


class StoreQueue[A <: Data]
(
  edge: AXI4EdgeParameters,
  reqGen: DMARequest, nInflight: Int,
  accReadGen: SRAMRead[A]
) extends BaseLoadStoreQueue(reqGen, nInflight) {

  val aw = IO(Decoupled(new AXI4BundleAW(edge.bundle)))
  val w = IO(Decoupled(new AXI4BundleW(edge.bundle)))
  val b = IO(Flipped(Decoupled(new AXI4BundleB(edge.bundle))))
  val accRead = IO(accReadGen.cloneType)

  val awPtr = Counter(nInflight)
  val rPtr = Counter(nInflight)

  val awEntry = entries(awPtr.value)

  aw.valid := entryValid(awPtr.value)
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
  require(accRead.data.getWidth % beatBits == 0)
  val nBeats = accRead.data.getWidth / beatBits

  // sram read
  val rBeatCnt = RegInit(0.U(log2Up(nBeats).W))
  val rLast = rBeatCnt === (nBeats - 1).U
  val writeQueue = Module(new Queue(UInt(beatBits.W), entries = 2, pipe = true))
  DelayedAssert(!writeQueue.io.enq.valid || writeQueue.io.enq.ready)

  val rEntry = entries(rPtr.value)
  accRead.valid := entryValid(rPtr.value) && Mux(writeQueue.io.enq.valid,
    writeQueue.io.count === 0.U,
    writeQueue.io.enq.ready
  ) && (rEntry.rRepeat > rEntry.req.repeat)
  accRead.addr := rEntry.req.sramAddr
  writeQueue.io.enq.valid := RegNext(accRead.fire, init = false.B)
  writeQueue.io.enq.bits := accRead.data.asTypeOf(Vec(nBeats, UInt(beatBits.W)))(RegEnable(rBeatCnt, accRead.fire))
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

  // currently the request size must be equal to the size of accumulator row size
  DelayedAssert(!io.req.fire || io.req.bits.size === (accRead.data.asUInt.getWidth / 8).U)
}


class LoadQueue[E <: Data]
(
  edge: AXI4EdgeParameters, reqGen: DMARequest,
  nInflight: Int, spadWriteGen: SRAMWrite[E]
) extends BaseLoadStoreQueue(reqGen, nInflight) {

  val ar = IO(Decoupled(new AXI4BundleAR(edge.bundle)))
  val r = IO(Flipped(Decoupled(new AXI4BundleR(edge.bundle))))
  val spadWrite = IO(spadWriteGen.cloneType)

  val maxTransfer = edge.slave.maxTransfer
  val maxBurstLen = maxTransfer / edge.slave.beatBytes
  when(io.req.fire) {
    assert(io.req.bits.size.take(log2Up(edge.slave.beatBytes)) === 0.U, "Currently size must be a multiple of the beat size")
    assert(io.req.bits.size <= maxTransfer.U, "Each request must be able to fit in a single AXI4 burst transfer")
  }

  val arPtr = Counter(nInflight)
  val rPtr = Counter(nInflight)

  // read address
  val arEntry = entries(arPtr.value)
  ar.valid := entryValid(arPtr.value) && arEntry.req.repeat =/= 0.U
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
  val rBeatCnt = RegInit(0.U(log2Up(maxBurstLen).W))
  val nLanes = spadWrite.data.getWidth / (edge.slave.beatBytes * 8)
  require(spadWrite.data.getWidth % (edge.slave.beatBytes * 8) == 0)
  val wData = VecInit(Seq.fill(nLanes){ r.bits.data }).asTypeOf(spadWrite.data)
  val wMask = VecInit(
    (0 until nLanes).map(i => i.U === rBeatCnt).flatMap(b =>
      Seq.fill(spadWrite.mask.length / nLanes)(b)
    )
  )
  spadWrite.valid := r.valid
  spadWrite.addr := entries(rPtr.value).req.sramAddr
  spadWrite.data := wData
  spadWrite.mask := wMask
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
