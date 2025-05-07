package msaga.isa

import chisel3._
import ISA.Constants._

class FenceInstruction extends NBytesBundle(4) with HasInstructionType {
  val matrix = Bool()
  val dma = Bool()
  val stop = Bool()
  val _pad = padOpt(I_TYPE_BITS + 3)
  checkWidth()
}
