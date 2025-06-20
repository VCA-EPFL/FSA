package msaga.frontend

import chisel3._
import chisel3.util._
import msaga.MSAGAModule
import msaga.isa.{DMAInstruction, FenceInstruction, MatrixInstruction}
import msaga.isa.ISA._
import msaga.isa.ISA.Constants.I_TYPE_BITS
import org.chipsalliance.cde.config.Parameters

class InstructionMerger(n: Int) extends Module {
  val io = IO(new Bundle {
    val in = Flipped(Decoupled(UInt(32.W)))
    val out = Decoupled(UInt((n * 32).W))
    val inflight = Output(Bool())
  })
  val buf = Reg(Vec(n, UInt(32.W)))
  val cnt = RegInit(0.U(n.U.getWidth.W))

  val w_addr = Mux(io.out.fire, 0.U, cnt)

  when(io.in.fire) {
    buf(w_addr) := io.in.bits
    cnt := cnt + 1.U
  }
  when(io.out.fire) {
    cnt := io.in.fire.asUInt
  }
  io.out.valid := cnt === n.U
  io.out.bits := buf.asUInt
  io.in.ready := cnt < n.U || cnt === n.U && io.out.fire
  io.inflight := cnt =/= 0.U && !io.out.valid
}

class Decoder(memAddrWidth: Int)(implicit p: Parameters) extends MSAGAModule {
  val io = IO(new Bundle {
    val in = Flipped(Decoupled(UInt(32.W)))
    val outMx = Decoupled(new MatrixInstruction(SPAD_ROW_ADDR_WIDTH, ACC_ROW_ADDR_WIDTH))
    val outDMA = Decoupled(new DMAInstruction(SRAM_ROW_ADDR_WIDTH, memAddrWidth))
    val outFence = Decoupled(new FenceInstruction)
  })
  val mx = Module(new InstructionMerger(3))
  val dma = Module(new InstructionMerger(4))
  assert(!(mx.io.inflight && dma.io.inflight))

  val instType = io.in.bits.head(I_TYPE_BITS)
  val first = !mx.io.inflight && !dma.io.inflight
  val selMx = first && instType === InstTypes.MATRIX.U || mx.io.inflight
  val selDma = first && instType === InstTypes.DMA.U || dma.io.inflight
  val selFence = first && instType === InstTypes.FENCE.U

  mx.io.in.valid := selMx && io.in.valid
  mx.io.in.bits := io.in.bits
  dma.io.in.valid := selDma && io.in.valid
  dma.io.in.bits := io.in.bits

  io.outMx.valid := mx.io.out.valid
  io.outMx.bits := mx.io.out.bits.asTypeOf(io.outMx.bits)
  mx.io.out.ready := io.outMx.ready

  io.outDMA.valid := dma.io.out.valid
  io.outDMA.bits := dma.io.out.bits.asTypeOf(io.outDMA.bits)
  dma.io.out.ready := io.outDMA.ready

  io.outFence.valid := selFence && io.in.valid
  io.outFence.bits := io.in.bits.asTypeOf(io.outFence.bits)

  io.in.ready := Mux(selMx, mx.io.in.ready, Mux(selDma, dma.io.in.ready, io.outFence.ready))
}
