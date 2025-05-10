package msaga.isa

import chisel3._
import ISA.Constants._

class DMAInstructionHeader extends NBytesBundle(4) with HasInstructionType {
  val func = UInt(DMA_FUNC_BITS.W)
  val consumerSemId = UInt(SEM_ID_BITS.W)
  val consumerSemValue = UInt(SEM_VALUE_BITS.W)
  val producerSemId = UInt(SEM_ID_BITS.W)
  val producerSemValue = UInt(SEM_VALUE_BITS.W)
  val repeat = UInt(DMA_REPEAT_BITS.W)
  val _pad = padOpt(I_TYPE_BITS + DMA_FUNC_BITS + 2 * (SEM_ID_BITS + SEM_VALUE_BITS) + DMA_REPEAT_BITS)
  checkWidth()
}

class DMAInstructionSRAM(val addrWidth: Int) extends NBytesBundle(4) with HasAddr {
  val stride = SInt(SRAM_STRIDE_BITS.W)
  val isAccum = Bool()
  val _pad = padOpt(SRAM_MAX_ADDR_BITS + SRAM_STRIDE_BITS + 1)
  override def maxAddrWidth: Int = SRAM_MAX_ADDR_BITS
  checkWidth()
}

class DMAInstructionMem(val addrWidth: Int) extends NBytesBundle(8) with HasAddr {
  val stride = SInt(MEM_STRIDE_BITS.W)
  val size = UInt(DMA_SIZE_BITS.W)
  val _pad = padOpt(MEM_MAX_ADDR_BITS + MEM_STRIDE_BITS + DMA_SIZE_BITS)
  override def maxAddrWidth: Int = MEM_MAX_ADDR_BITS
  checkWidth()
}

class DMAInstruction(sramAddrWidth: Int, memAddrWidth: Int) extends NBytesBundle(16) {
  val mem = new DMAInstructionMem(memAddrWidth)
  val sram = new DMAInstructionSRAM(sramAddrWidth)
  val header = new DMAInstructionHeader
  checkWidth()
}
