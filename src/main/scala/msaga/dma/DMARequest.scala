package msaga.dma

import chisel3._
import chisel3.util._
import msaga.isa.HasSemaphore
import msaga.isa.ISA.Constants._


class DMARequest(val sramAddrWidth: Int, val memAddrWidth: Int) extends Bundle with HasSemaphore {
  val memAddr = UInt(memAddrWidth.W)
  val memStride = SInt(MEM_STRIDE_BITS.W)
  val sramAddr = UInt(sramAddrWidth.W)
  val sramStride = SInt(SRAM_STRIDE_BITS.W)
  val repeat = UInt(DMA_REPEAT_BITS.W)
  val size = UInt(DMA_SIZE_BITS.W)
  val isLoad = Bool()
}

// Partition the request into `n` parts across `repeat`
class RequestPartitioner(reqGen: DMARequest, n: Int) extends Module {
  val io = IO(new Bundle {
    val in = Flipped(Decoupled(reqGen))
    val out = Decoupled(Vec(n, reqGen))
  })

  if (n == 1) {
    io.out.valid := io.in.valid
    io.out.bits.head := io.in.bits
    io.in.ready := io.out.ready
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

    for ((out, req) <- io.out.bits.zip(reqs)) {
      out := req
    }
    io.out.valid := valid
    io.in.ready := !valid || io.out.ready
    when(io.out.fire) {
      valid := false.B
    }
    when(io.in.fire) {
      valid := true.B
    }
  }
}
