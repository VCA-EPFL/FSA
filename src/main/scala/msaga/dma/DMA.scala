package msaga.dma

import freechips.rocketchip.amba.axi4._
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.subsystem._
import org.chipsalliance.cde.config._
import org.chipsalliance.diplomacy.lazymodule._
import testchipip.soc.SubsystemInjector
import chisel3._
import chisel3.util._
import freechips.rocketchip.tilelink._
import msaga.SRAMWrite
import msaga.isa.DMAInstruction
import msaga.isa.ISA.Constants._
import msaga.frontend.SemaphoreWrite
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


class LoadHandler[E <: Data](edge: AXI4EdgeParameters, reqGen: DMARequest, nInflight: Int, spadWriteGen: SRAMWrite[E]) extends Module {
  val io = IO(new Bundle {
    val ar = DecoupledIO(new AXI4BundleAR(edge.bundle))
    val r = Flipped(DecoupledIO(new AXI4BundleR(edge.bundle)))
    val req = Flipped(Decoupled(reqGen))
    val semWrite = Decoupled(new SemaphoreWrite)
    val spadWrite = spadWriteGen.cloneType
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
  io.inst.ready := dmaReq.ready

  val partitioner = Module(new RequestPartitioner(chiselTypeOf(dmaReq.bits), nPorts))
  partitioner.io.in <> dmaReq

  val loadHandlers = io.spadWrite.zip(partitioner.io.out.zip(node.out)).map { case (spad, (req, (axi, edge))) =>
    val loadHandler = Module(new LoadHandler(edge, chiselTypeOf(dmaReq.bits), dmaLoadInflight, spad))
    loadHandler.io.req <> req
    loadHandler.io.r <> axi.r
    axi.ar <> loadHandler.io.ar
    spad <> loadHandler.io.spadWrite
    loadHandler
  }

  val loadSemWriteReady = Cat(loadHandlers.map(_.io.semWrite.valid)).andR
  loadHandlers.foreach(_.io.semWrite.ready := loadSemWriteReady)
  io.semaphoreWrite.bits := loadHandlers.head.io.semWrite.bits
  io.semaphoreWrite.valid := loadHandlers.head.io.semWrite.fire
  when(io.semaphoreWrite.fire) {
    /* since we only use 1 axi id, responses arrives in order
       requests in different load handlers finish in order
     */
    for (i <- 1 until nPorts) {
      DelayedAssert(loadHandlers(i).io.semWrite.bits.asUInt === io.semaphoreWrite.bits.asUInt)
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