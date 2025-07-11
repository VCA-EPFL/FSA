package fsa.isa

import chisel3._
import ISA.Constants._

class DMAInstructionHeader extends NBytesBundle(4) with HasInstructionType with HasSemaphore {
  val func = UInt(DMA_FUNC_BITS.W)
  val repeat = UInt(DMA_REPEAT_BITS.W)
  val _pad = padOpt(I_TYPE_BITS + semBits + DMA_FUNC_BITS + DMA_REPEAT_BITS)
  checkWidth()
}

class DMAInstructionSRAM(val addrWidth: Int) extends NBytesBundle(4) with HasAddr {
  val stride = SInt(SRAM_STRIDE_BITS.W)
  val isAccum = Bool()
  val mem_stride1 = UInt(MEM_STRIDE_1_BITS.W)
  val _pad = padOpt(SRAM_MAX_ADDR_BITS + SRAM_STRIDE_BITS + 1 + MEM_STRIDE_1_BITS)
  override def maxAddrWidth: Int = SRAM_MAX_ADDR_BITS
  checkWidth()
}

class DMAInstructionMem(val addrWidth: Int) extends NBytesBundle(8) with HasAddr {
  val stride2 = UInt(MEM_STRIDE_2_BITS.W)
  val size = UInt(DMA_SIZE_BITS.W)
  val _pad = padOpt(MEM_MAX_ADDR_BITS + MEM_STRIDE_2_BITS + DMA_SIZE_BITS)
  override def maxAddrWidth: Int = MEM_MAX_ADDR_BITS
  checkWidth()
}

class DMAInstruction(sramAddrWidth: Int, memAddrWidth: Int) extends NBytesBundle(16) {
  val mem = new DMAInstructionMem(memAddrWidth)
  val sram = new DMAInstructionSRAM(sramAddrWidth)
  val header = new DMAInstructionHeader
  checkWidth()

  def getStride: SInt = (sram.mem_stride1 ## mem.stride2).asSInt
}
