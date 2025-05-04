package msaga

import freechips.rocketchip.util.UIntIsOneOf
import chisel3._
import chisel3.util._

object ISA {
  def SRAM_MAX_ADDR_WIDTH = 24
  def STRIDE_MAX_WIDTH = 8
  def RS_WIDTH = 64
  object Func {
    def LOAD_STATIONARY = 0.U
    def ATTENTION_SCORE_COMPUTE = 1.U
    def ATTENTION_VALUE_COMPUTE = 2.U
    def ATTENTION_LSE_NORM_SCALE = 3.U
    def ATTENTION_LSE_NORM = 4.U

    def wait_for_accumulator(func: UInt): Bool = func.isOneOf(ATTENTION_LSE_NORM_SCALE, ATTENTION_LSE_NORM)
  }
}

class Instruction extends Bundle {
  val funct7 = UInt(7.W)
  val rs1 = UInt(ISA.RS_WIDTH.W)
  val rs2 = UInt(ISA.RS_WIDTH.W)
}

class MatrixRs(addrWidth: Int) extends Bundle {
  // --------------------------------------------------
  val _pad1 = UInt((ISA.SRAM_MAX_ADDR_WIDTH - addrWidth).W)
  val addr = UInt(addrWidth.W)
  val stride = SInt(ISA.STRIDE_MAX_WIDTH.W)
  // ---------------------------------------------------
  val reverseInput = Bool()
  val reverseOutput = Bool()
  val delayOutput = Bool()
  val _pad2 = UInt((ISA.RS_WIDTH - 32 - 3).W)
  require(this.getWidth == ISA.RS_WIDTH)
}
