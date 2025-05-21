package msaga.isa

import chisel3._
import ISA.Constants._

class MatrixInstructionHeader extends NBytesBundle(4) with HasInstructionType with HasSemaphore {
  val func = UInt(MX_FUNC_BITS.W)
  val waitPrevAcc = Bool()
  val _pad = padOpt(I_TYPE_BITS + semBits + MX_FUNC_BITS + 1)
  checkWidth()
}

class MatrixInstructionSpad(val addrWidth: Int) extends NBytesBundle(4) with HasAddr {
  val stride = SInt(SPAD_STRIDE_BITS.W)
  val revInput = Bool()
  val revOutput = Bool()
  val delayOutput = Bool()
  val _pad = padOpt(SPAD_MAX_ADDR_BITS + SPAD_STRIDE_BITS + 3)
  override def maxAddrWidth: Int = SPAD_MAX_ADDR_BITS
  checkWidth()
}

class MatrixInstructionAcc(val addrWidth: Int) extends NBytesBundle(4) with HasAddr {
  val stride = SInt(ACC_STRIDE_BITS.W)
  val zero = Bool()
  val _pad = padOpt(ACC_MAX_ADDR_BITS + ACC_STRIDE_BITS + 1)
  override def maxAddrWidth: Int = ACC_MAX_ADDR_BITS
  checkWidth()
}

class MatrixInstruction(spAddrWidth: Int, accAddrWidth: Int) extends NBytesBundle(12) {
  val acc = new MatrixInstructionAcc(accAddrWidth)
  val spad = new MatrixInstructionSpad(spAddrWidth)
  val header = new MatrixInstructionHeader
  checkWidth()
}
