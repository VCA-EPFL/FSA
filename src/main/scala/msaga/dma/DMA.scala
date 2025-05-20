package msaga.dma

import org.chipsalliance.diplomacy.lazymodule._
import freechips.rocketchip.amba.axi4._
import freechips.rocketchip.diplomacy._
import org.chipsalliance.cde.config._
import chisel3._
import chisel3.util._
import msaga.{SRAMRead, SRAMWrite}
import msaga.isa.DMAInstruction
import msaga.isa.ISA.Constants._
import msaga.frontend.SemaphoreWrite
import msaga.isa.ISA.DMAFunc
import msaga.utils.DelayedAssert

class DMARequest(val sramAddrWidth: Int, val memAddrWidth: Int) extends Bundle {
  val memAddr = UInt(memAddrWidth.W)
  val memStride = SInt(MEM_STRIDE_BITS.W)
  val sramAddr = UInt(sramAddrWidth.W)
  val sramStride = SInt(SRAM_STRIDE_BITS.W)
  val repeat = UInt(DMA_REPEAT_BITS.W)
  val size = UInt(DMA_SIZE_BITS.W)
  val dstSemId = UInt(SEM_ID_BITS.W)
  val dstSemValue = UInt(SEM_VALUE_BITS.W)
  val isLoad = Bool()
}

// Partition the request into `n` parts across `repeat`
class RequestPartitioner(reqGen: DMARequest, n: Int) extends Module {
  val io = IO(new Bundle {
    val in = Flipped(Decoupled(reqGen))
    val out = Vec(n, Decoupled(reqGen))
  })

  if (n == 1) {
    io.out.head <> io.in
  } else {
    val reqs = Reg(Vec(n, reqGen))
    val valid = RegInit(false.B)

    val initialRepeatCnt = (io.in.bits.repeat >> log2Up(n).U).asUInt
    val remainingRepeatCnt = io.in.bits.repeat.take(log2Up(n))
    for ((req, i) <- reqs.zipWithIndex) {
      val addRem = remainingRepeatCnt > i.U
      val repeatCnt = Mux(addRem,
        initialRepeatCnt + 1.U,
        initialRepeatCnt
      )
      val addrIncr = (initialRepeatCnt * (i.U +& Mux(addRem, i.U, remainingRepeatCnt))).zext
      when(io.in.fire) {
        req := io.in.bits
        req.repeat := repeatCnt
        req.memAddr := (io.in.bits.memAddr.asSInt + addrIncr * io.in.bits.memStride).asUInt
        req.sramAddr := (io.in.bits.sramAddr.asSInt + addrIncr * io.in.bits.sramStride).asUInt
      }
    }

    val outValid = valid && Cat(io.out.map(_.ready)).andR
    for ((out, req) <- io.out.zip(reqs)) {
      out.bits := req
      /* to make semaphore write condition check easier,
         we always allocate an entry for each port even if repeat is 0
       */
      out.valid := outValid
    }
    io.in.ready := !valid || outValid
    when(outValid) {
      valid := false.B
    }
    when(io.in.fire) {
      valid := true.B
    }
  }
}

class StoreHandler[A <: Data]
(
  edge: AXI4EdgeParameters,
  reqGen: DMARequest, nInflight: Int,
  accReadGen: SRAMRead[A]
) extends Module {

  val io = IO(new Bundle {
    val aw = Decoupled(new AXI4BundleAW(edge.bundle))
    val w = Decoupled(new AXI4BundleW(edge.bundle))
    val b = Flipped(Decoupled(new AXI4BundleB(edge.bundle)))
    val req = Flipped(Decoupled(reqGen))
    val accRead = accReadGen.cloneType
    val semWrite = Decoupled(new SemaphoreWrite)
    val busy = Output(Bool())
  })

  class Entry extends Bundle {
    val req = reqGen.cloneType
    val rRepeat = UInt(DMA_REPEAT_BITS.W)
  }
  val entries = Reg(Vec(nInflight, new Entry))

  val entryValid = RegInit(VecInit(Seq.fill(nInflight)(false.B)))
  val enqPtr = RegInit(0.U(log2Up(nInflight).W))
  val awPtr = RegInit(0.U(log2Up(nInflight).W))
  val rPtr = RegInit(0.U(log2Up(nInflight).W))
  val releasePtr = RegInit(0.U(log2Up(nInflight).W))

  val (aw, w, b) = (io.aw, io.w, io.b)

  require(isPow2(nInflight))

  val awEntry = entries(awPtr)

  aw.valid := entryValid(awPtr)
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
      awPtr := awPtr + 1.U
    }
  }

  val beatBits = edge.slave.beatBytes * 8
  require(io.accRead.data.getWidth % beatBits == 0)
  val nBeats = io.accRead.data.getWidth / beatBits

  // sram read
  val rBeatCnt = RegInit(0.U(log2Up(nBeats).W))
  val rLast = rBeatCnt === (nBeats - 1).U
  val writeQueue = Module(new Queue(UInt(beatBits.W), entries = 2, pipe = true))
  DelayedAssert(!writeQueue.io.enq.valid || writeQueue.io.enq.ready)

  val rEntry = entries(rPtr)
  io.accRead.valid := entryValid(rPtr) && Mux(writeQueue.io.enq.valid,
    writeQueue.io.count === 0.U,
    writeQueue.io.enq.ready
  ) && (rEntry.rRepeat > rEntry.req.repeat)
  io.accRead.addr := rEntry.req.sramAddr
  writeQueue.io.enq.valid := RegNext(io.accRead.fire, init = false.B)
  writeQueue.io.enq.bits := io.accRead.data.asTypeOf(Vec(nBeats, UInt(beatBits.W)))(rBeatCnt)
  when(io.accRead.fire) {
    rBeatCnt := Mux(rLast, 0.U, rBeatCnt + 1.U)
    when(rLast) {
      rEntry.rRepeat := rEntry.rRepeat - 1.U
      rEntry.req.sramAddr := (rEntry.req.sramAddr.asSInt + rEntry.req.sramStride).asUInt
      when(rEntry.rRepeat === 1.U) {
        rPtr := rPtr + 1.U
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

  // release
  val releaseEntry = entries(releasePtr)
  io.semWrite.valid := entryValid(releasePtr) && releaseEntry.rRepeat === 0.U
  io.semWrite.bits.semaphoreId := releaseEntry.req.dstSemId
  io.semWrite.bits.semaphoreValue := releaseEntry.req.dstSemValue
  when(io.semWrite.fire) {
    releasePtr := releasePtr + 1.U
    entryValid(releasePtr) := false.B
  }

  // enq request
  when(io.req.fire) {
    entries(enqPtr).req := io.req.bits
    entries(enqPtr).rRepeat := io.req.bits.repeat
    entryValid(enqPtr) := true.B
    enqPtr := enqPtr + 1.U
    // currently the request size must be equal to the size of accumulator row size
    DelayedAssert(io.req.bits.size === (io.accRead.data.asUInt.getWidth / 8).U)
  }
  io.req.ready := !entryValid(enqPtr)
  io.busy := entryValid(releasePtr)
  dontTouch(io)
}


class LoadHandler[E <: Data](edge: AXI4EdgeParameters, reqGen: DMARequest, nInflight: Int, spadWriteGen: SRAMWrite[E]) extends Module {
  val io = IO(new Bundle {
    val ar = Decoupled(new AXI4BundleAR(edge.bundle))
    val r = Flipped(Decoupled(new AXI4BundleR(edge.bundle)))
    val req = Flipped(Decoupled(reqGen))
    val semWrite = Decoupled(new SemaphoreWrite)
    val spadWrite = spadWriteGen.cloneType
    val busy = Output(Bool())
  })

  require(isPow2(nInflight))

  // val maxTransfer = (1 << AXI4Parameters.lenBits) * edge.slave.beatBytes
  val maxTransfer = edge.slave.maxTransfer
  val maxBurstLen = maxTransfer / edge.slave.beatBytes
  when(io.req.fire) {
    assert(io.req.bits.size.take(log2Up(edge.slave.beatBytes)) === 0.U, "Currently size must be a multiple of the beat size")
    assert(io.req.bits.size <= maxTransfer.U, "Each request must be able to fit in a single AXI4 burst transfer")
  }

  val (ar, r) = (io.ar, io.r)

  class Entry extends Bundle {
    val req = reqGen.cloneType
    // duplicate the repeat field for counting the read response
    val rRepeat = UInt(DMA_REPEAT_BITS.W)
  }

  val entries = Reg(Vec(nInflight, new Entry))

  val entryValid = RegInit(VecInit(Seq.fill(nInflight)(false.B)))
  val enqPtr = RegInit(0.U(log2Up(nInflight).W))
  val arPtr = RegInit(0.U(log2Up(nInflight).W))
  val rPtr = RegInit(0.U(log2Up(nInflight).W))
  val releasePtr = RegInit(0.U(log2Up(nInflight).W))

  // read address
  val arEntry = entries(arPtr)
  ar.valid := entryValid(arPtr) && arEntry.req.repeat =/= 0.U
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
      arPtr := arPtr + 1.U
    }
  }

  // read data
  val rBeatCnt = RegInit(0.U(log2Up(maxBurstLen).W))
  val nLanes = io.spadWrite.data.getWidth / (edge.slave.beatBytes * 8)
  require(io.spadWrite.data.getWidth % (edge.slave.beatBytes * 8) == 0)
  val wData = VecInit(Seq.fill(nLanes){ r.bits.data }).asTypeOf(io.spadWrite.data)
  val wMask = VecInit(
    (0 until nLanes).map(i => i.U === rBeatCnt).flatMap(b =>
      Seq.fill(io.spadWrite.mask.length / nLanes)(b)
    )
  )
  io.spadWrite.valid := r.valid
  io.spadWrite.addr := entries(rPtr).req.sramAddr
  io.spadWrite.data := wData
  io.spadWrite.mask := wMask
  r.ready := io.spadWrite.ready

  when(r.fire) {
    val rEntry = entries(rPtr)
    assert(r.bits.id === 0.U, "Currently only one ID is supported")
    assert(r.bits.resp === AXI4Parameters.RESP_OKAY, "Currently only OKAY response is supported")
    rBeatCnt := Mux(r.bits.last, 0.U, rBeatCnt + 1.U)
    when(r.bits.last) {
      rEntry.rRepeat := rEntry.rRepeat - 1.U
      rEntry.req.sramAddr := (rEntry.req.sramAddr.asSInt + rEntry.req.sramStride).asUInt
      when(rEntry.rRepeat === 1.U) {
        rPtr := rPtr + 1.U
      }
    }
  }

  // semaphore update
  val releaseEntry = entries(releasePtr)
  io.semWrite.valid := entryValid(releasePtr) && releaseEntry.rRepeat === 0.U
  io.semWrite.bits.semaphoreId := releaseEntry.req.dstSemId
  io.semWrite.bits.semaphoreValue := releaseEntry.req.dstSemValue
  when(io.semWrite.fire) {
    releasePtr := releasePtr + 1.U
    entryValid(releasePtr) := false.B
  }

  // enq request
  when(io.req.fire) {
    entries(enqPtr).req := io.req.bits
    entries(enqPtr).rRepeat := io.req.bits.repeat
    entryValid(enqPtr) := true.B
    enqPtr := enqPtr + 1.U
  }
  io.req.ready := !entryValid(enqPtr)
  io.busy := entryValid(releasePtr)

  dontTouch(io)
  dontTouch(rBeatCnt)

}

class DMAImpl[E <: Data, A <: Data](outer: DMA[E, A]) extends LazyModuleImp(outer) {
  val node = outer.node
  val nPorts = node.out.size
  val memAddrWidth = node.out.map(_._2.bundle.addrBits).max
  val sramAddrWidth = outer.sramAddrWidth
  val (dmaLoadInflight, dmaStoreInflight) = (outer.dmaLoadInflight, outer.dmaStoreInflight)

  val io = IO(new Bundle {
    val inst = Flipped(Decoupled(new DMAInstruction(sramAddrWidth, memAddrWidth)))
    val semaphoreWrite = Valid(new SemaphoreWrite)
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
  dmaReq.bits.dstSemId := io.inst.bits.header.producerSemId
  dmaReq.bits.dstSemValue := io.inst.bits.header.producerSemValue
  dmaReq.bits.isLoad := io.inst.bits.header.func === DMAFunc.LD_SRAM
  io.inst.ready := dmaReq.ready

  val partitioner = Module(new RequestPartitioner(chiselTypeOf(dmaReq.bits), nPorts))
  partitioner.io.in <> dmaReq

  val (loadHandlers, storeHandlers) = io.spadWrite.zip(io.accRead).zip(partitioner.io.out).zip(node.out).map{
    case (((spad, acc), req), (axi, edge)) =>
      val loadHandler = Module(new LoadHandler(edge, chiselTypeOf(dmaReq.bits), dmaLoadInflight, spad))
      val storeHandler = Module(new StoreHandler(edge, chiselTypeOf(dmaReq.bits), dmaStoreInflight, acc))
      loadHandler.io.req.valid := req.valid && req.bits.isLoad
      loadHandler.io.req.bits := req.bits
      storeHandler.io.req.valid := req.valid && !req.bits.isLoad
      storeHandler.io.req.bits := req.bits
      req.ready := Mux(req.bits.isLoad, loadHandler.io.req.ready, storeHandler.io.req.ready)
      axi.ar <> loadHandler.io.ar
      axi.aw <> storeHandler.io.aw
      loadHandler.io.r <> axi.r
      axi.w <> storeHandler.io.w
      storeHandler.io.b <> axi.b
      spad <> loadHandler.io.spadWrite
      acc <> storeHandler.io.accRead
      (loadHandler, storeHandler)
  }.unzip

  io.busy := Cat(
    dmaReq.valid,
    partitioner.io.out.head.valid,
    Cat(loadHandlers.map(_.io.busy)).orR,
    Cat(storeHandlers.map(_.io.busy)).orR
  ).orR

  val loadSemWrite = Wire(Decoupled(new SemaphoreWrite))
  val storeSemWrite = Wire(Decoupled(new SemaphoreWrite))

  loadSemWrite.valid := Cat(loadHandlers.map(_.io.semWrite.valid)).andR
  loadSemWrite.bits := loadHandlers.head.io.semWrite.bits
  loadHandlers.foreach(_.io.semWrite.ready := loadSemWrite.fire)

  storeSemWrite.valid := Cat(storeHandlers.map(_.io.semWrite.valid)).andR
  storeSemWrite.bits := storeHandlers.head.io.semWrite.bits
  storeHandlers.foreach(_.io.semWrite.ready := storeSemWrite.fire)

  val arb = Module(new Arbiter(new SemaphoreWrite, 2))
  arb.io.in(0) <> loadSemWrite
  arb.io.in(1) <> storeSemWrite
  arb.io.out.ready := true.B
  io.semaphoreWrite.valid := arb.io.out.valid
  io.semaphoreWrite.bits := arb.io.out.bits


  when(loadSemWrite.fire) {
    /* since we only use 1 axi id, responses arrives in order
       requests in different load handlers finish in order
     */
    for (i <- 1 until nPorts) {
      DelayedAssert(loadHandlers(i).io.semWrite.bits.asUInt === io.semaphoreWrite.bits.asUInt)
    }
  }
  when(storeSemWrite.fire) {
    for (i <- 1 until nPorts) {
      DelayedAssert(storeHandlers(i).io.semWrite.bits.asUInt === io.semaphoreWrite.bits.asUInt)
    }
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